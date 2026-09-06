#!/usr/bin/env python3
"""
Strict vs lenient physics-decoder comparison on the corrected held-out benchmark.

Paired sampling: each generative (u, z) draw is decoded in both modes so attempt counts
are identical per event; retained counts and fractions differ by decoder only.

No training. Frozen stage-u / stage-z checkpoints only.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.data_u import load_all_u_data, load_standardization_stats, merge_player_constants  # noqa: E402
from sbi_forward_sim.src.data_z import load_all_z_data_native_circular  # noqa: E402
from sbi_forward_sim.src.feature_contract_z import load_feature_contract_z  # noqa: E402
from sbi_forward_sim.src.heldout_forward_inference import (  # noqa: E402
    build_corrected_eligibility_audit,
    merge_baseline_master_row,
    u_encoder_inputs_from_row,
    z_base_x_num_from_row,
    z_categoricals_from_row,
)
from sbi_forward_sim.src.models_u import SharedMixtureGaussianVonMisesUNet  # noqa: E402
from sbi_forward_sim.src.models_z import ConditionalHybridGaussVonMisesMixtureZ  # noqa: E402
from sbi_forward_sim.src.physics_decoder import decode_bip_batch  # noqa: E402
from sbi_forward_sim.src.physics_decoder_contract import (  # noqa: E402
    DECODER_G_NUMERIC_REQUIRED,
    DECODER_G_OPTIONAL,
    DECODER_P_REQUIRED,
    DecoderInputs,
    DecoderInputError,
    validate_decoder_context_g,
)
from sbi_forward_sim.src.pipeline_heldout_sample import (  # noqa: E402
    patch_z_x_num_with_u,
    sample_u_shared_gaussian_vm,
    sample_z_hybrid_native_batched,
)
from sbi_forward_sim.src.target_transform_z import wrap_deg  # noqa: E402


def _row_float(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    v = row[col]
    return float(v) if pd.notna(v) else float("nan")


def _git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_MMC2_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        return None


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_u_model(cfg: dict, vocabs: dict, device: torch.device) -> SharedMixtureGaussianVonMisesUNet:
    mcfg = cfg["model"]
    num_f = len(cfg["numeric_features"]) + len(cfg["player_constant_features"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return SharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_mixture=int(mcfg["n_mixture_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=float(mcfg.get("kappa_max", 120.0)),
    ).to(device)


def _build_z_model(cfg: dict, vocabs: dict, device: torch.device) -> ConditionalHybridGaussVonMisesMixtureZ:
    mcfg = cfg["model"]
    fc = load_feature_contract_z(cfg, Path(__file__).resolve().parents[1])
    num_f = len(fc["x_numeric_zscore_column_order"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesMixtureZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=int(mcfg["n_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
    ).to(device)


def _g_dict_from_master_row(row: pd.Series) -> dict[str, Any]:
    g: dict[str, Any] = {}
    for k in DECODER_G_NUMERIC_REQUIRED:
        if k not in row.index:
            raise DecoderInputError(f"context master row missing decoder g key {k!r}")
        g[k] = float(row[k])
    has_deg = "spin_axis_deg" in row.index and pd.notna(row.get("spin_axis_deg"))
    if has_deg:
        g["spin_axis_deg"] = float(row["spin_axis_deg"])
    else:
        for k in ("spin_axis_sin", "spin_axis_cos"):
            if k not in row.index:
                raise DecoderInputError(f"missing {k} for spin bundle")
            g[k] = float(row[k])
    for ok in DECODER_G_OPTIONAL:
        if ok in row.index and pd.notna(row.get(ok)):
            g[ok] = float(row[ok])
    validate_decoder_context_g(g)
    return g


def _p_dict_from_row(row: pd.Series) -> dict[str, float]:
    p: dict[str, float] = {}
    for k in DECODER_P_REQUIRED:
        if k not in row.index:
            raise DecoderInputError(f"missing player constant {k!r}")
        p[k] = float(row[k])
    return p


def _empirical_crps(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n < 2:
        return float("nan")
    e1 = np.mean(np.abs(s - y))
    e2 = np.mean(np.abs(s.reshape(-1, 1) - s.reshape(1, -1)))
    return float(e1 - 0.5 * e2)


def _pit_value(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n == 0:
        return float("nan")
    r = np.sum(s < y) + 0.5 * np.sum(s == y)
    return float(np.clip(r / n, 1e-6, 1 - 1e-6))


def _coverage_width(samples: np.ndarray, y: float, qlo: float, qhi: float) -> tuple[bool, float]:
    s = np.asarray(samples, dtype=np.float64).ravel()
    if len(s) < 2:
        return False, float("nan")
    lo, hi = np.quantile(s, [qlo, qhi])
    inside = bool(lo <= y <= hi)
    return inside, float(hi - lo)


def _prepare_stats_u(project_root: Path, u_cfg: dict[str, Any], master: pd.DataFrame) -> dict[str, Any]:
    stats_u = copy.deepcopy(load_standardization_stats(project_root / u_cfg["paths"]["standardization_stats"]))
    ut = pd.read_parquet(project_root / u_cfg["paths"]["u_train"])
    ut = merge_player_constants(ut, master)
    for c in u_cfg["player_constant_features"]:
        if c not in stats_u:
            v = pd.to_numeric(ut[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats_u[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
    return stats_u


def _prepare_stats_z(
    project_root: Path, z_cfg: dict[str, Any], z_fc: dict[str, Any], master: pd.DataFrame
) -> dict[str, Any]:
    stats = copy.deepcopy(load_standardization_stats(project_root / z_cfg["paths"]["standardization_stats"]))
    zt = pd.read_parquet(project_root / z_cfg["paths"]["z_train"])
    zt = merge_player_constants(zt, master)
    for t in z_fc["targets"]:
        if t not in stats:
            v = pd.to_numeric(zt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
    for c in z_fc["player_constant_features"]:
        if c not in stats:
            v = pd.to_numeric(zt[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
    return stats


def _qdict(s: pd.Series, qs: list[float]) -> dict[str, float]:
    s = s.dropna().to_numpy()
    if len(s) == 0:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q*100)}"] = float(np.quantile(s, q))
    return out


def _collect_metrics_and_plots(
    out_sub: Path,
    mode_label: str,
    df_ev: pd.DataFrame,
    df_draw: pd.DataFrame,
    df_fail: pd.DataFrame,
    events_eval: list[int],
    u_run: Path,
    z_run: Path,
    baseline_path: Path,
    master_path: Path,
    audit_summary: dict[str, Any],
    target_n: int,
    max_attempts_per_event: int,
    seed: int,
    sampling_design: str,
) -> dict[str, Any]:
    plots_dir = Path(out_sub) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    qlist = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for tgt in ("EV", "LA", "SA"):
        means, medians, stds = [], [], []
        qcols: dict[str, list] = {f"pred_{tgt}_{k}": [] for k in [f"q{int(q*100)}" for q in qlist]}
        for _, er in df_ev.iterrows():
            sub = df_draw.loc[df_draw["event_id"] == er["event_id"], tgt]
            arr = sub.to_numpy(dtype=np.float64)
            if len(arr) == 0:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(np.nan)
                for qc in qcols:
                    qcols[qc].append(np.nan)
                continue
            means.append(float(np.mean(arr)))
            medians.append(float(np.quantile(arr, 0.5)))
            stds.append(float(np.std(arr, ddof=0)))
            qd = _qdict(sub, qlist)
            for k in qd:
                qcols[f"pred_{tgt}_{k}"].append(qd[k])
        df_ev[f"pred_{tgt}_mean"] = means
        df_ev[f"pred_{tgt}_median"] = medians
        df_ev[f"pred_{tgt}_std"] = stds
        for k, v in qcols.items():
            df_ev[k] = v

    metrics: dict[str, Any] = {
        "decoder_mode": mode_label,
        "n_events": int(len(events_eval)),
        "target_retained_draws_per_event": target_n,
        "max_attempts_per_event_cap": max_attempts_per_event,
        "sampling_design": sampling_design,
        "total_attempted_joint_draws": int(df_ev["attempts_total"].sum()),
        "total_retained_draws": int(df_ev["retained_draws"].sum()),
        "mean_retained_fraction": float(df_ev["retained_fraction"].mean()),
        "n_events_target_retained_met": int(df_ev["target_retained_met"].sum()),
        "n_events_target_retained_not_met": int((~df_ev["target_retained_met"]).sum()),
        "mean_attempts_to_finish": float(df_ev["attempts_total"].mean()),
        "frozen_stage_u_run": str(u_run),
        "frozen_stage_z_run": str(z_run),
        "observed_y_source": str(baseline_path),
        "master_path": str(master_path),
        "held_out_event_universe": audit_summary.get("definition", ""),
        "audit_eligible_count": int(audit_summary["n_eligible_for_forward_pipeline"]),
        "rng_seed_base": seed,
        "note_LA": "LA = decoder launch_angle_deg; linear error in degrees.",
        "note_SA": "SA spray deg; wrapped error where noted.",
    }

    pits_ev, pits_la, pits_sa = [], [], []
    crps_ev, crps_la, crps_sa = [], [], []
    cov50 = {t: [] for t in ("EV", "LA", "SA")}
    cov80 = {t: [] for t in ("EV", "LA", "SA")}
    cov90 = {t: [] for t in ("EV", "LA", "SA")}
    width50 = {t: [] for t in ("EV", "LA", "SA")}
    width80 = {t: [] for t in ("EV", "LA", "SA")}
    width90 = {t: [] for t in ("EV", "LA", "SA")}
    mae_mean = {t: [] for t in ("EV", "LA", "SA")}
    rmse_mean = {t: [] for t in ("EV", "LA", "SA")}
    bias_mean = {t: [] for t in ("EV", "LA", "SA")}

    for _, er in df_ev.iterrows():
        eid = int(er["event_id"])
        sub = df_draw.loc[df_draw["event_id"] == eid]
        for tgt, obs, pit_list, crp_list, use_wrap in [
            ("EV", er["observed_EV"], pits_ev, crps_ev, False),
            ("LA", er["observed_LA"], pits_la, crps_la, False),
            ("SA", er["observed_SA"], pits_sa, crps_sa, True),
        ]:
            samps = sub[tgt].to_numpy(dtype=np.float64)
            pred_mean = float(er[f"pred_{tgt}_mean"]) if np.isfinite(er[f"pred_{tgt}_mean"]) else np.nan
            err_mean = float(pred_mean) - float(obs)
            if use_wrap:
                err_mean = float(wrap_deg(float(pred_mean) - float(obs)))
            mae_mean[tgt].append(abs(err_mean))
            rmse_mean[tgt].append(err_mean**2)
            bias_mean[tgt].append(err_mean)
            if len(samps) > 0:
                pit_list.append(_pit_value(samps, float(obs)))
                crp_list.append(_empirical_crps(samps, float(obs)))
                i50, w50 = _coverage_width(samps, float(obs), 0.25, 0.75)
                cov50[tgt].append(float(i50))
                i80, w80 = _coverage_width(samps, float(obs), 0.1, 0.9)
                cov80[tgt].append(float(i80))
                i90, w90 = _coverage_width(samps, float(obs), 0.05, 0.95)
                cov90[tgt].append(float(i90))
                width50[tgt].append(w50)
                width80[tgt].append(w80)
                width90[tgt].append(w90)
            else:
                pit_list.append(np.nan)
                crp_list.append(np.nan)
                cov50[tgt].append(np.nan)
                cov80[tgt].append(np.nan)
                cov90[tgt].append(np.nan)
                width50[tgt].append(np.nan)
                width80[tgt].append(np.nan)
                width90[tgt].append(np.nan)

    metrics["point_accuracy_event_mean_prediction"] = {
        "MAE": {t: float(np.nanmean(mae_mean[t])) for t in mae_mean},
        "RMSE": {t: float(np.sqrt(np.nanmean(rmse_mean[t]))) for t in rmse_mean},
        "bias": {t: float(np.nanmean(bias_mean[t])) for t in bias_mean},
    }
    metrics["probabilistic_sample_based"] = {
        "mean_CRPS": {
            "EV": float(np.nanmean(crps_ev)),
            "LA": float(np.nanmean(crps_la)),
            "SA": float(np.nanmean(crps_sa)),
        },
        "coverage_rate_marginal_sample_central": {
            "EV": {
                "50": float(np.nanmean(cov50["EV"])),
                "80": float(np.nanmean(cov80["EV"])),
                "90": float(np.nanmean(cov90["EV"])),
            },
            "LA": {
                "50": float(np.nanmean(cov50["LA"])),
                "80": float(np.nanmean(cov80["LA"])),
                "90": float(np.nanmean(cov90["LA"])),
            },
            "SA": {
                "50": float(np.nanmean(cov50["SA"])),
                "80": float(np.nanmean(cov80["SA"])),
                "90": float(np.nanmean(cov90["SA"])),
            },
        },
        "mean_interval_width_50": {
            "EV": float(np.nanmean(width50["EV"])),
            "LA": float(np.nanmean(width50["LA"])),
            "SA": float(np.nanmean(width50["SA"])),
        },
        "mean_interval_width_80": {
            "EV": float(np.nanmean(width80["EV"])),
            "LA": float(np.nanmean(width80["LA"])),
            "SA": float(np.nanmean(width80["SA"])),
        },
        "mean_interval_width_90": {
            "EV": float(np.nanmean(width90["EV"])),
            "LA": float(np.nanmean(width90["LA"])),
            "SA": float(np.nanmean(width90["SA"])),
        },
    }
    metrics["pit_histogram_counts"] = {
        "EV": np.histogram(pits_ev, bins=20, range=(0, 1))[0].tolist(),
        "LA": np.histogram(pits_la, bins=20, range=(0, 1))[0].tolist(),
        "SA": np.histogram(pits_sa, bins=20, range=(0, 1))[0].tolist(),
    }

    if mode_label == "lenient" and len(df_draw) and "strict_diagnostic_flags" in df_draw.columns:
        flag_ct: Counter[str] = Counter()
        for s in df_draw["strict_diagnostic_flags"].astype(str):
            for part in str(s).split(";"):
                t = part.strip()
                if t:
                    flag_ct[t] += 1
        metrics["retained_draws_strict_diagnostic_flag_totals"] = dict(flag_ct.most_common(500))

    w_mae_sa, w_sq_sa = [], []
    for _, er in df_ev.iterrows():
        sub = df_draw.loc[df_draw["event_id"] == er["event_id"], "SA"]
        pm = float(er["pred_SA_mean"])
        if len(sub) > 0:
            w_mae_sa.append(abs(wrap_deg(pm - float(er["observed_SA"]))))
            w_sq_sa.append(wrap_deg(pm - float(er["observed_SA"])) ** 2)
    metrics["point_accuracy_wrapped_SA_mean"] = {
        "MAE": float(np.nanmean(w_mae_sa)) if w_mae_sa else float("nan"),
        "RMSE": float(np.sqrt(np.nanmean(w_sq_sa))) if w_sq_sa else float("nan"),
    }

    out_sub = Path(out_sub)
    df_ev.to_parquet(out_sub / "predictive_summary_by_event.parquet", index=False)
    df_ev.to_csv(out_sub / "predictive_summary_by_event.csv", index=False)
    df_draw.to_parquet(out_sub / "predictive_draws_retained.parquet", index=False)
    df_draw.to_csv(out_sub / "predictive_draws_retained.csv", index=False)
    if len(df_fail):
        df_fail.to_parquet(out_sub / "predictive_draws_failed.parquet", index=False)
        df_fail.to_csv(out_sub / "predictive_draws_failed.csv", index=False)
    (out_sub / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sum_rows: list[list[Any]] = [
        ["n_events", metrics["n_events"]],
        ["target_retained_draws_per_event", target_n],
        ["max_attempts_per_event_cap", max_attempts_per_event],
        ["mean_attempts_joint_draws_per_event", float(df_ev["attempts_total"].mean())],
        ["mean_retained_fraction", metrics["mean_retained_fraction"]],
        ["events_reaching_target_retained", metrics["n_events_target_retained_met"]],
    ]
    for tgt in ("EV", "LA", "SA"):
        sum_rows.append([f"MAE_mean_pred_{tgt}", metrics["point_accuracy_event_mean_prediction"]["MAE"][tgt]])
        sum_rows.append([f"RMSE_mean_pred_{tgt}", metrics["point_accuracy_event_mean_prediction"]["RMSE"][tgt]])
        sum_rows.append([f"bias_mean_pred_{tgt}", metrics["point_accuracy_event_mean_prediction"]["bias"][tgt]])
        sum_rows.append([f"CRPS_{tgt}", metrics["probabilistic_sample_based"]["mean_CRPS"][tgt]])
        for level in ("50", "80", "90"):
            sum_rows.append(
                [
                    f"coverage_{tgt}_{level}",
                    metrics["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"][tgt][level],
                ]
            )
    pd.DataFrame(sum_rows, columns=["metric", "value"]).to_csv(out_sub / "metrics_summary.csv", index=False)

    def _pit_plot(vals: list[float], name: str, title: str) -> None:
        v = [x for x in vals if np.isfinite(x)]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(v, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("PIT")
        fig.tight_layout()
        fig.savefig(plots_dir / f"pit_{name}.png", dpi=120)
        plt.close(fig)

    _pit_plot(pits_ev, "EV", f"PIT EV ({mode_label})")
    _pit_plot(pits_la, "LA", f"PIT LA ({mode_label})")
    _pit_plot(pits_sa, "SA", f"PIT SA ({mode_label})")

    for tgt, obs_col in [("EV", "observed_EV"), ("LA", "observed_LA"), ("SA", "observed_SA")]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(df_ev[obs_col], df_ev[f"pred_{tgt}_mean"], alpha=0.8, edgecolors="k", linewidths=0.5)
        mx = float(np.nanmax([df_ev[obs_col].max(), df_ev[f"pred_{tgt}_mean"].max()]))
        mn = float(np.nanmin([df_ev[obs_col].min(), df_ev[f"pred_{tgt}_mean"].min()]))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel(f"Observed {tgt}")
        ax.set_ylabel(f"Pred mean {tgt}")
        ax.set_title(mode_label)
        fig.tight_layout()
        fig.savefig(plots_dir / f"scatter_mean_vs_obs_{tgt}.png", dpi=120)
        plt.close(fig)

    cov_rates = metrics["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]
    nominal = [0.5, 0.8, 0.9]
    for tgt in ("EV", "LA", "SA"):
        fig, ax = plt.subplots(figsize=(4, 3))
        emp = [cov_rates[tgt]["50"], cov_rates[tgt]["80"], cov_rates[tgt]["90"]]
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w / 2, nominal, width=w, label="nominal", color="lightgray", edgecolor="k")
        ax.bar(x + w / 2, emp, width=w, label="empirical", color="steelblue", edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(["50%", "80%", "90%"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage rate")
        ax.set_title(f"{tgt} coverage — {mode_label}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"coverage_intervals_{tgt}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(df_ev["retained_fraction"].to_numpy(), bins=min(20, max(5, len(df_ev))), color="coral", edgecolor="black")
    ax.set_xlabel("Retained fraction (retained / attempts)")
    ax.set_ylabel("Count")
    ax.set_title(f"{mode_label}: retained fraction per event")
    fig.tight_layout()
    fig.savefig(plots_dir / "hist_retained_fraction.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(df_ev["attempts_total"].to_numpy(), bins=min(30, max(5, len(df_ev))), color="seagreen", edgecolor="black")
    ax.set_xlabel("Attempts (joint draws) per event")
    ax.set_ylabel("Count")
    ax.set_title(f"{mode_label}: attempts until stop")
    fig.tight_layout()
    fig.savefig(plots_dir / "hist_attempts_per_event.png", dpi=120)
    plt.close(fig)

    if len(df_fail) and "failure_reason" in df_fail.columns:
        vc = df_fail["failure_reason"].value_counts().head(25)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.plot(kind="bar", ax=ax, color="gray")
        ax.set_title(f"Decoder hard-fail reasons — {mode_label} (top 25)")
        fig.tight_layout()
        fig.savefig(plots_dir / "failure_reason_counts.png", dpi=120)
        plt.close(fig)

    return {
        "metrics": metrics,
        "pits": {"EV": pits_ev, "LA": pits_la, "SA": pits_sa},
        "cov": metrics["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"],
        "df_ev": df_ev,
        "df_fail": df_fail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--u-run-dir",
        type=Path,
        default=Path("outputs/p_u_given_g/20260407_193337Z"),
        help="Frozen stage-u run directory",
    )
    ap.add_argument(
        "--z-run-dir",
        type=Path,
        default=Path("outputs/p_z_given_u_g/20260407_185923Z"),
        help="Frozen stage-z run directory",
    )
    ap.add_argument("--seed", type=int, default=20260408)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--target-retained", type=int, default=100)
    ap.add_argument("--max-attempts-per-event", type=int, default=10000)
    ap.add_argument("--sample-batch-size", type=int, default=512)
    ap.add_argument("--decode-chunk-size", type=int, default=2048)
    ap.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on eligible events (debug); default = all eligible.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    target_n = int(args.target_retained)
    cap = int(args.max_attempts_per_event)
    batch_sz = max(1, int(args.sample_batch_size))
    decode_chunk = max(1, int(args.decode_chunk_size))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_root = (project_root / "outputs" / "heldout_decoder_mode_comparison" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cmp_plots = out_root / "comparison_plots"
    cmp_plots.mkdir(exist_ok=True)
    strict_dir = out_root / "strict"
    lenient_dir = out_root / "lenient"
    strict_dir.mkdir(exist_ok=True)
    lenient_dir.mkdir(exist_ok=True)
    (strict_dir / "plots").mkdir(exist_ok=True)
    (lenient_dir / "plots").mkdir(exist_ok=True)

    u_run = (project_root / args.u_run_dir).resolve()
    z_run = (project_root / args.z_run_dir).resolve()
    u_ckpt = _load_ckpt(u_run / "checkpoint.pt")
    z_ckpt = _load_ckpt(z_run / "checkpoint.pt")
    u_cfg = u_ckpt["config"]
    z_cfg = z_ckpt["config"]
    u_vocabs: dict = u_ckpt["vocabs"]
    z_vocabs: dict = z_ckpt["vocabs"]

    z_fc = load_feature_contract_z(z_cfg, project_root)
    z_num_order = list(z_fc["x_numeric_zscore_column_order"])
    master_path = project_root / u_cfg["paths"]["context_master"]
    master = pd.read_parquet(master_path)
    baseline_path = project_root / "data_processed" / "baseline_direct_y_test.parquet"
    baseline = pd.read_parquet(baseline_path)
    if "split" in baseline.columns:
        baseline = baseline.loc[baseline["split"].eq("test")].copy()

    stats_u = _prepare_stats_u(project_root, u_cfg, master)
    stats_z = _prepare_stats_z(project_root, z_cfg, z_fc, master)
    u_cols_tpl = ("v_ss_tilde", "a_tilde", "d_tilde")
    audit_df, audit_summary = build_corrected_eligibility_audit(
        baseline, master, stats_u, stats_z, u_cfg, z_num_order, u_vocabs, z_vocabs, u_cols_tpl
    )
    for sub in (strict_dir, lenient_dir):
        audit_df.to_csv(sub / "event_eligibility_audit.csv", index=False)
        audit_summary["generated_utc"] = stamp
        (sub / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    (out_root / "event_eligibility_audit.csv").write_text(audit_df.to_csv(index=False))
    (out_root / "audit_summary.json").write_text(json.dumps({**audit_summary, "generated_utc": stamp}, indent=2))

    need_m = list(DECODER_G_NUMERIC_REQUIRED) + ["event_id", "batter_name"]
    miss_m = [c for c in need_m if c not in master.columns]
    if miss_m:
        raise DecoderInputError(f"context master missing columns: {miss_m}")

    train_u, _, _, _ = load_all_u_data(u_cfg, project_root, vocabs_override=u_vocabs)
    train_z, _, _, _ = load_all_z_data_native_circular(z_cfg, project_root, vocabs_override=z_vocabs)

    events_eval = sorted(int(x) for x in audit_df.loc[audit_df["eligible_for_pipeline"], "event_id"].tolist())
    if not events_eval:
        raise RuntimeError("No eligible held-out events.")
    if args.max_events is not None:
        events_eval = events_eval[: max(0, int(args.max_events))]
    if not events_eval:
        raise RuntimeError("No events after --max-events filter.")

    u_model = _build_u_model(u_cfg, u_vocabs, device)
    u_model.load_state_dict(u_ckpt["model_state"])
    u_model.eval()
    z_model = _build_z_model(z_cfg, z_vocabs, device)
    z_model.load_state_dict(z_ckpt["model_state"])
    z_model.eval()

    T_va = float(u_ckpt["temperature_T_va"])
    T_ang_u = float(u_ckpt["temperature_T_ang"])
    kappa_max_u = float(u_ckpt.get("kappa_max", u_cfg["model"].get("kappa_max", 120.0)))
    T_gauss_z = float(z_ckpt["temperature_T_gauss"])
    T_ang_z = float(z_ckpt["temperature_T_ang"])
    T_gx = z_ckpt.get("temperature_T_gauss_x")
    T_gy = z_ckpt.get("temperature_T_gauss_y")
    T_gx_f = float(T_gx) if T_gx is not None else None
    T_gy_f = float(T_gy) if T_gy is not None else None

    tm_u = torch.tensor(train_u.target_means, dtype=torch.float32, device=device)
    ts_u = torch.tensor(train_u.target_stds, dtype=torch.float32, device=device)
    gm_z = torch.tensor(train_z.gauss_means, dtype=torch.float32, device=device)
    gs_z = torch.tensor(train_z.gauss_stds, dtype=torch.float32, device=device)

    g_gen_u = torch.Generator(device=device)
    g_gen_z = torch.Generator(device=device)

    u_cols = u_cols_tpl

    rows_draw_s: list[dict[str, Any]] = []
    rows_draw_l: list[dict[str, Any]] = []
    rows_fail_s: list[dict[str, Any]] = []
    rows_fail_l: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []

    def _draw_ok(out_row: pd.Series) -> bool:
        if not bool(out_row.get("admissible", False)):
            return False
        for k in ("EV", "LA", "SA"):
            v = _row_float(out_row, k)
            if not np.isfinite(v):
                return False
        return True

    n_ev_tot = len(events_eval)
    for ev_i, eid in enumerate(events_eval):
        if ev_i % 50 == 0:
            print(f"Events progress: {ev_i}/{n_ev_tot} (event_id={eid})", flush=True)
        mrow = master.loc[master["event_id"] == eid].iloc[0]
        brow = baseline.loc[baseline["event_id"] == eid].iloc[0]
        row = merge_baseline_master_row(brow, master)
        g_dec = _g_dict_from_master_row(mrow)
        p_dec = _p_dict_from_row(mrow)
        batter = str(mrow["batter_name"])
        obs_ev = float(brow["EV"])
        obs_la = float(brow["LA"])
        obs_sa = float(brow["SA"])

        x_u_np, cat_u_np = u_encoder_inputs_from_row(row, stats_u, u_vocabs, u_cfg)
        xu = torch.from_numpy(x_u_np).to(device)
        catu = {k: torch.from_numpy(cat_u_np[k]).long().to(device) for k in sorted(u_vocabs.keys())}
        base_z = z_base_x_num_from_row(row, z_num_order, stats_z, u_cols)
        cat_z_np = z_categoricals_from_row(row, z_vocabs)
        catz_template = {k: int(cat_z_np[k][0]) for k in sorted(z_vocabs.keys())}

        rng_e = np.random.default_rng(zlib.adler32(bytes(f"{args.seed}:{eid}", "utf-8")) & 0xFFFFFFFF)
        n_strict = 0
        n_lenient = 0
        attempts = 0
        fails_s: dict[str, int] = {}
        fails_l: dict[str, int] = {}
        draw_idx = 0

        while (n_strict < target_n or n_lenient < target_n) and attempts < cap:
            b = min(batch_sz, cap - attempts)
            if b <= 0:
                break
            sub_seed = int(rng_e.integers(0, 2**31 - 1))
            g_gen_u.manual_seed(sub_seed)
            g_gen_z.manual_seed(sub_seed + 17)

            u_pack = sample_u_shared_gaussian_vm(
                u_model,
                xu,
                catu,
                n_samples=b,
                T_va=T_va,
                T_ang=T_ang_u,
                kappa_max=kappa_max_u,
                target_means=tm_u,
                target_stds=ts_u,
                generator=g_gen_u,
            )
            u_np = np.stack(
                [
                    u_pack["v_ss_tilde"].cpu().numpy(),
                    u_pack["a_tilde"].cpu().numpy(),
                    u_pack["d_tilde"].cpu().numpy(),
                ],
                axis=1,
            )
            x_z_np = patch_z_x_num_with_u(base_z, z_num_order, u_cols, u_np, stats_z)
            x_z = torch.from_numpy(x_z_np).to(device)
            catz = {
                k: torch.full((b,), catz_template[k], device=device, dtype=torch.long)
                for k in sorted(z_vocabs.keys())
            }
            z_pack = sample_z_hybrid_native_batched(
                z_model,
                x_z,
                catz,
                n_samples_per_row=1,
                T_gauss=T_gauss_z,
                T_gauss_x=T_gx_f,
                T_gauss_y=T_gy_f,
                T_ang=T_ang_z,
                gauss_means=gm_z,
                gauss_stds=gs_z,
                generator=g_gen_z,
            )
            x_samp = z_pack["x"][:, 0].cpu().numpy()
            ey_samp = z_pack["e_y_star"][:, 0].cpu().numpy()
            psi_samp = z_pack["psi_deg"][:, 0].cpu().numpy()
            th_samp = z_pack["theta_deg"][:, 0].cpu().numpy()

            inps: list[DecoderInputs] = []
            u_rows: list[dict[str, float]] = []
            z_rows: list[dict[str, float]] = []
            for j in range(b):
                uj = {
                    "v_ss_tilde": float(u_np[j, 0]),
                    "a_tilde": float(u_np[j, 1]),
                    "d_tilde": float(u_np[j, 2]),
                }
                zj = {
                    "x": float(x_samp[j]),
                    "e_y_star": float(ey_samp[j]),
                    "psi_deg": float(psi_samp[j]),
                    "theta_deg": float(th_samp[j]),
                }
                u_rows.append(uj)
                z_rows.append(zj)
                inps.append(
                    DecoderInputs(
                        event_id=int(eid),
                        batter_name=batter,
                        g=g_dec,
                        p=p_dec,
                        u=uj,
                        z=zj,
                    )
                )
            attempts += b

            # strict
            for j0 in range(0, b, decode_chunk):
                j1 = min(j0 + decode_chunk, b)
                chunk_inps = inps[j0:j1]
                try:
                    df_bs = decode_bip_batch(chunk_inps, admissibility="strict")
                except DecoderInputError as ex:
                    for j in range(j0, j1):
                        if n_strict >= target_n:
                            break
                        rows_fail_s.append(
                            {
                                "event_id": eid,
                                "draw_idx": draw_idx + j,
                                "failure_reason": f"batch_validate:{ex}",
                                **{f"u_{k}": v for k, v in u_rows[j].items()},
                                **{f"z_{k}": v for k, v in z_rows[j].items()},
                            }
                        )
                        fails_s["batch_validate"] = fails_s.get("batch_validate", 0) + 1
                    continue
                for rel, j in enumerate(range(j0, j1)):
                    if n_strict >= target_n:
                        break
                    out = df_bs.iloc[rel]
                    if not _draw_ok(out):
                        reason = str(out.get("failure_reason") or "inadmissible")
                        rows_fail_s.append(
                            {
                                "event_id": eid,
                                "draw_idx": draw_idx + j,
                                "failure_reason": reason,
                                **{f"u_{k}": v for k, v in u_rows[j].items()},
                                **{f"z_{k}": v for k, v in z_rows[j].items()},
                            }
                        )
                        fails_s[reason] = fails_s.get(reason, 0) + 1
                        continue
                    n_strict += 1
                    uj, zj = u_rows[j], z_rows[j]
                    rows_draw_s.append(
                        {
                            "event_id": eid,
                            "batter_name": batter,
                            "retained_idx": n_strict - 1,
                            "global_draw_idx": draw_idx + j,
                            "u_v_ss_tilde": uj["v_ss_tilde"],
                            "u_a_tilde": uj["a_tilde"],
                            "u_d_tilde": uj["d_tilde"],
                            "z_x": zj["x"],
                            "z_psi_deg": zj["psi_deg"],
                            "z_e_y_star": zj["e_y_star"],
                            "z_theta_deg": zj["theta_deg"],
                            "EV": _row_float(out, "EV"),
                            "LA": _row_float(out, "LA"),
                            "SA": _row_float(out, "SA"),
                            "e_x": _row_float(out, "e_x"),
                            "omega_plus_rad_s": _row_float(out, "omega_plus_rad_s"),
                            "strict_diagnostic_flags": out.get("strict_diagnostic_flags") or "",
                        }
                    )

            # lenient
            for j0 in range(0, b, decode_chunk):
                j1 = min(j0 + decode_chunk, b)
                chunk_inps = inps[j0:j1]
                try:
                    df_bl = decode_bip_batch(chunk_inps, admissibility="lenient")
                except DecoderInputError as ex:
                    for j in range(j0, j1):
                        if n_lenient >= target_n:
                            break
                        rows_fail_l.append(
                            {
                                "event_id": eid,
                                "draw_idx": draw_idx + j,
                                "failure_reason": f"batch_validate:{ex}",
                                **{f"u_{k}": v for k, v in u_rows[j].items()},
                                **{f"z_{k}": v for k, v in z_rows[j].items()},
                            }
                        )
                        fails_l["batch_validate"] = fails_l.get("batch_validate", 0) + 1
                    continue
                for rel, j in enumerate(range(j0, j1)):
                    if n_lenient >= target_n:
                        break
                    out = df_bl.iloc[rel]
                    if not _draw_ok(out):
                        reason = str(out.get("failure_reason") or "inadmissible")
                        rows_fail_l.append(
                            {
                                "event_id": eid,
                                "draw_idx": draw_idx + j,
                                "failure_reason": reason,
                                **{f"u_{k}": v for k, v in u_rows[j].items()},
                                **{f"z_{k}": v for k, v in z_rows[j].items()},
                            }
                        )
                        fails_l[reason] = fails_l.get(reason, 0) + 1
                        continue
                    n_lenient += 1
                    uj, zj = u_rows[j], z_rows[j]
                    flags = str(out.get("strict_diagnostic_flags") or "")
                    rows_draw_l.append(
                        {
                            "event_id": eid,
                            "batter_name": batter,
                            "retained_idx": n_lenient - 1,
                            "global_draw_idx": draw_idx + j,
                            "u_v_ss_tilde": uj["v_ss_tilde"],
                            "u_a_tilde": uj["a_tilde"],
                            "u_d_tilde": uj["d_tilde"],
                            "z_x": zj["x"],
                            "z_psi_deg": zj["psi_deg"],
                            "z_e_y_star": zj["e_y_star"],
                            "z_theta_deg": zj["theta_deg"],
                            "EV": _row_float(out, "EV"),
                            "LA": _row_float(out, "LA"),
                            "SA": _row_float(out, "SA"),
                            "e_x": _row_float(out, "e_x"),
                            "omega_plus_rad_s": _row_float(out, "omega_plus_rad_s"),
                            "strict_diagnostic_flags": flags,
                        }
                    )

            draw_idx += b

        rf_s = n_strict / max(attempts, 1)
        rf_l = n_lenient / max(attempts, 1)
        event_summaries.append(
            {
                "event_id": eid,
                "batter_name": batter,
                "split": "test",
                "observed_EV": obs_ev,
                "observed_LA": obs_la,
                "observed_SA": obs_sa,
                "attempts_total": attempts,
                "target_retained": target_n,
                "retained_draws_strict": n_strict,
                "retained_draws_lenient": n_lenient,
                "retained_fraction_strict": rf_s,
                "retained_fraction_lenient": rf_l,
                "target_retained_met_strict": bool(n_strict >= target_n),
                "target_retained_met_lenient": bool(n_lenient >= target_n),
                "failure_reason_counts_strict_json": json.dumps(fails_s),
                "failure_reason_counts_lenient_json": json.dumps(fails_l),
            }
        )

    df_ev_joint = pd.DataFrame(event_summaries)
    df_ev_joint.to_csv(out_root / "event_level_retention_strict_vs_lenient.csv", index=False)

    df_ev_s = df_ev_joint.drop(
        columns=[
            "retained_draws_lenient",
            "retained_fraction_lenient",
            "target_retained_met_lenient",
            "failure_reason_counts_lenient_json",
        ]
    ).rename(
        columns={
            "retained_draws_strict": "retained_draws",
            "retained_fraction_strict": "retained_fraction",
            "target_retained_met_strict": "target_retained_met",
            "failure_reason_counts_strict_json": "failure_reason_counts_json",
        }
    )
    df_ev_s["attempts_total"] = df_ev_joint["attempts_total"]
    df_ev_l = df_ev_joint.drop(
        columns=[
            "retained_draws_strict",
            "retained_fraction_strict",
            "target_retained_met_strict",
            "failure_reason_counts_strict_json",
        ]
    ).rename(
        columns={
            "retained_draws_lenient": "retained_draws",
            "retained_fraction_lenient": "retained_fraction",
            "target_retained_met_lenient": "target_retained_met",
            "failure_reason_counts_lenient_json": "failure_reason_counts_json",
        }
    )
    df_ev_l["attempts_total"] = df_ev_joint["attempts_total"]

    df_draw_s = pd.DataFrame(rows_draw_s)
    df_draw_l = pd.DataFrame(rows_draw_l)
    df_fail_s = pd.DataFrame(rows_fail_s)
    df_fail_l = pd.DataFrame(rows_fail_l)

    sampling_design = (
        "paired_joint_draws: each sampled (u,z) decoded in strict and lenient; "
        "attempts_total is identical per event for both modes; retention differs."
    )

    res_s = _collect_metrics_and_plots(
        strict_dir,
        "strict",
        df_ev_s,
        df_draw_s,
        df_fail_s,
        events_eval,
        u_run,
        z_run,
        baseline_path,
        master_path,
        audit_summary,
        target_n,
        cap,
        args.seed,
        sampling_design,
    )
    res_l = _collect_metrics_and_plots(
        lenient_dir,
        "lenient",
        df_ev_l,
        df_draw_l,
        df_fail_l,
        events_eval,
        u_run,
        z_run,
        baseline_path,
        master_path,
        audit_summary,
        target_n,
        cap,
        args.seed,
        sampling_design,
    )

    ms, ml = res_s["metrics"], res_l["metrics"]
    for name, sub, r in (
        ("strict", strict_dir, res_s),
        ("lenient", lenient_dir, res_l),
    ):
        manifest = {
            "timestamp_utc": stamp,
            "git_hash": _git_hash(),
            "decoder_mode": name,
            "stage_u_checkpoint": str(u_run / "checkpoint.pt"),
            "stage_z_checkpoint": str(z_run / "checkpoint.pt"),
            "n_events_evaluated": len(events_eval),
            "target_retained_per_event": target_n,
            "max_attempts_per_event_cap": cap,
            "sample_batch_size": batch_sz,
            "decode_chunk_size": decode_chunk,
            "sampling_design": sampling_design,
            "outputs_dir": str(sub),
            "held_out_event_universe": audit_summary.get("definition", ""),
        }
        (sub / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- Comparison plots ---
    for tgt in ("EV", "LA", "SA"):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
        for ax, label, pits in zip(axes, ("strict", "lenient"), (res_s["pits"][tgt], res_l["pits"][tgt]), strict=True):
            v = [x for x in pits if np.isfinite(x)]
            ax.hist(v, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
            ax.set_title(label)
            ax.set_xlabel("PIT")
        axes[0].set_ylabel("count")
        fig.suptitle(f"PIT {tgt}: strict vs lenient")
        fig.tight_layout()
        fig.savefig(cmp_plots / f"pit_compare_{tgt}.png", dpi=120)
        plt.close(fig)

    for tgt in ("EV", "LA", "SA"):
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(3)
        w = 0.35
        nom = [0.5, 0.8, 0.9]
        s_emp = [res_s["cov"][tgt]["50"], res_s["cov"][tgt]["80"], res_s["cov"][tgt]["90"]]
        l_emp = [res_l["cov"][tgt]["50"], res_l["cov"][tgt]["80"], res_l["cov"][tgt]["90"]]
        ax.bar(x - w, nom, width=w, label="nominal", color="lightgray", edgecolor="k")
        ax.bar(x, s_emp, width=w, label="strict", color="#4c72b0", edgecolor="k")
        ax.bar(x + w, l_emp, width=w, label="lenient", color="#55a868", edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(["50%", "80%", "90%"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("coverage")
        ax.legend(fontsize=8)
        ax.set_title(f"{tgt} central interval coverage")
        fig.tight_layout()
        fig.savefig(cmp_plots / f"coverage_compare_{tgt}.png", dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    for ax, label, fr in zip(axes, ("strict", "lenient"), (df_ev_s["retained_fraction"], df_ev_l["retained_fraction"])):
        ax.hist(fr.to_numpy(), bins=20, color="coral", edgecolor="black")
        ax.set_title(label)
        ax.set_xlabel("retained fraction")
    axes[0].set_ylabel("count")
    fig.suptitle("Retained fraction per event (paired attempts)")
    fig.tight_layout()
    fig.savefig(cmp_plots / "retained_fraction_compare.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        df_ev_s["retained_fraction"].to_numpy(),
        df_ev_l["retained_fraction"].to_numpy(),
        alpha=0.5,
        edgecolors="k",
        linewidths=0.3,
    )
    mx = max(float(df_ev_s["retained_fraction"].max()), float(df_ev_l["retained_fraction"].max()))
    ax.plot([0, mx], [0, mx], "r--", lw=1)
    ax.set_xlabel("strict retained fraction")
    ax.set_ylabel("lenient retained fraction")
    ax.set_title("Per-event retention: strict vs lenient")
    fig.tight_layout()
    fig.savefig(cmp_plots / "retained_fraction_scatter_strict_vs_lenient.png", dpi=120)
    plt.close(fig)

    if len(df_fail_s) or len(df_fail_l):
        top_s = (
            df_fail_s["failure_reason"].value_counts().head(15)
            if len(df_fail_s) and "failure_reason" in df_fail_s.columns
            else pd.Series(dtype=int)
        )
        top_l = (
            df_fail_l["failure_reason"].value_counts().head(15)
            if len(df_fail_l) and "failure_reason" in df_fail_l.columns
            else pd.Series(dtype=int)
        )
        all_reasons = sorted(set(top_s.index.tolist()) | set(top_l.index.tolist()))
        if all_reasons:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(all_reasons))))
            y = np.arange(len(all_reasons))
            w = 0.35
            s_ct = [int(top_s.get(r, 0)) for r in all_reasons]
            l_ct = [int(top_l.get(r, 0)) for r in all_reasons]
            ax.barh(y - w / 2, s_ct, height=w, label="strict", color="#4c72b0", edgecolor="k")
            ax.barh(y + w / 2, l_ct, height=w, label="lenient", color="#55a868", edgecolor="k")
            ax.set_yticks(y)
            ax.set_yticklabels(all_reasons, fontsize=7)
            ax.set_xlabel("count (failed draws)")
            ax.legend()
            ax.set_title("Hard failure reasons (intersection of top categories)")
            fig.tight_layout()
            fig.savefig(cmp_plots / "failure_reason_compare.png", dpi=120)
            plt.close(fig)

    top_manifest = {
        "timestamp_utc": stamp,
        "output_root": str(out_root),
        "strict_dir": str(strict_dir),
        "lenient_dir": str(lenient_dir),
        "comparison_plots": str(cmp_plots),
        "sampling_design": sampling_design,
        "max_attempts_per_event": cap,
        "target_retained": target_n,
        "n_events_evaluated": len(events_eval),
        "n_audit_eligible": int(audit_summary["n_eligible_for_forward_pipeline"]),
        "max_events_filter": int(args.max_events) if args.max_events is not None else None,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(top_manifest, indent=2), encoding="utf-8")

    comparison = {"strict": ms, "lenient": ml}
    (out_root / "comparison_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # Report
    report_path = project_root / "reports" / "heldout_decoder_mode_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(m: dict, k1: str, k2: str | None = None) -> str:
        d = m[k1] if k2 is None else m[k1][k2]
        return json.dumps(d, indent=2)

    n_audit_elig = int(audit_summary["n_eligible_for_forward_pipeline"])
    n_eval = len(events_eval)

    bias_s = ms["point_accuracy_event_mean_prediction"]["bias"]
    bias_l = ml["point_accuracy_event_mean_prediction"]["bias"]
    crp_s = ms["probabilistic_sample_based"]["mean_CRPS"]
    crp_l = ml["probabilistic_sample_based"]["mean_CRPS"]
    cov_s = ms["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]
    cov_l = ml["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]

    def _closer_to_nominal(c_s: dict, c_l: dict, nom: float, level: str) -> str:
        es = abs(float(c_s[level]) - nom)
        el = abs(float(c_l[level]) - nom)
        if el < es - 0.02:
            return "lenient"
        if es < el - 0.02:
            return "strict"
        return "similar"

    cov_50_bits = ", ".join(
        f"{tgt} 50%: {_closer_to_nominal(cov_s[tgt], cov_l[tgt], 0.5, '50')}"
        for tgt in ("EV", "LA", "SA")
    )
    rec_parts = [
        f"**Retention:** Lenient mean retained fraction **{ml['mean_retained_fraction']:.4f}** vs strict **{ms['mean_retained_fraction']:.4f}** "
        f"on paired generative attempts (mean attempts per event **{float(df_ev_joint['attempts_total'].mean()):.1f}**, cap {cap}). "
        f"Events reaching {target_n} retained: strict **{ms['n_events_target_retained_met']}/{n_eval}**, lenient **{ml['n_events_target_retained_met']}/{n_eval}**.",
        "",
        f"**EV bias:** strict {bias_s['EV']:.4f}, lenient {bias_l['EV']:.4f} (smaller |bias| is better).",
        f"**Mean CRPS:** EV strict {crp_s['EV']:.4f} vs lenient {crp_l['EV']:.4f}; "
        f"LA {crp_s['LA']:.4f} vs {crp_l['LA']:.4f}; SA {crp_s['SA']:.4f} vs {crp_l['SA']:.4f}.",
        "",
        f"**Coverage (which mode is closer to nominal at 50% central):** {cov_50_bits}. See `comparison_plots/coverage_compare_*.png`.",
        "",
    ]
    if ml["mean_retained_fraction"] > ms["mean_retained_fraction"] + 0.01:
        rec_parts.append(
            "**Usable sample counts:** Lenient retains strictly more paired draws; for forward generative benchmarking and posterior diagnostics "
            "that need ~100 samples per event, lenient is operationally preferable when strict leaves holes below the target."
        )
    else:
        rec_parts.append("**Usable sample counts:** Retention improvement from lenient mode is modest on this run.")

    if abs(bias_l["EV"]) <= abs(bias_s["EV"]) and (crp_l["EV"] <= crp_s["EV"] or np.isnan(crp_l["EV"])):
        rec_parts.append(
            "**Judgment:** Lenient decoding is **not a different generative model** — same draws — but it **materially relaxes inverse-style rejection** "
            "so predictive summaries reflect more of the frozen p(u,z|·) mass mapped through kinematics. "
            "**Keep strict** for production inverse-style parity and **use lenient** for forward-pipeline / calibration tests where you need full retained samples; "
            "compare both when auditing whether rejection gates distort benchmarks."
        )
    else:
        rec_parts.append(
            "**Judgment:** Compare CRPS and bias tables closely; if lenient moves bias or coverage without improving CRPS, interpret extra retained mass as "
            "**off-manifold kinematics** (see `strict_diagnostic_flags` on lenient draws). **Keep both modes:** strict for production gates, lenient for forward stress-testing."
        )

    recommendation_md = "\n".join(rec_parts)

    lines = [
        "# Held-out benchmark: strict vs lenient physics decoder",
        "",
        "## Held-out event universe",
        "",
        f"**Definition (unchanged):** {audit_summary.get('definition', '')}",
        "",
        f"- Eligible events (audit): **{n_audit_elig}** (`eligible_for_pipeline` in `event_eligibility_audit.csv`).",
        f"- **Events evaluated in this run:** **{n_eval}**" + (" (subset via `--max-events`)" if n_eval < n_audit_elig else "") + ".",
        f"- Baseline: `data_processed/baseline_direct_y_test.parquet` (official `split == test` when present).",
        f"- Context master: `{master_path.name}`.",
        "",
        "## Frozen checkpoints (not modified)",
        "",
        f"- Stage-u: `{u_run}`",
        f"- Stage-z: `{z_run}`",
        "",
        "## Sampling and retention",
        "",
        f"- **Design:** {sampling_design}",
        f"- **Target retained draws per event:** {target_n}",
        f"- **Hard cap on joint attempts per event:** {cap}",
        f"- **Per-mode attempts:** identical per event (paired decoding). **Retained fractions** are "
        f"(`retained`/`attempts`); when both modes reach the target, both hit 100 retained so fractions match even though **which** draws are kept usually differs.",
        "",
        "### Summary",
        "",
        f"- Events with 100 retained (strict): **{ms['n_events_target_retained_met']}** / {len(events_eval)}",
        f"- Events with 100 retained (lenient): **{ml['n_events_target_retained_met']}** / {len(events_eval)}",
        f"- Mean joint attempts per event: **{float(df_ev_joint['attempts_total'].mean()):.1f}**",
        f"- Mean retained fraction strict: **{ms['mean_retained_fraction']:.4f}**",
        f"- Mean retained fraction lenient: **{ml['mean_retained_fraction']:.4f}**",
        "",
        "## Predictive metrics",
        "",
        "### Strict",
        "",
        "```json",
        _fmt(ms, "point_accuracy_event_mean_prediction"),
        "```",
        "",
        "```json",
        '"CRPS": ' + json.dumps(ms["probabilistic_sample_based"]["mean_CRPS"], indent=2),
        "```",
        "",
        "```json",
        '"coverage": '
        + json.dumps(ms["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"], indent=2),
        "```",
        "",
        "### Lenient",
        "",
        "```json",
        _fmt(ml, "point_accuracy_event_mean_prediction"),
        "```",
        "",
        "```json",
        '"CRPS": ' + json.dumps(ml["probabilistic_sample_based"]["mean_CRPS"], indent=2),
        "```",
        "",
        "```json",
        '"coverage": '
        + json.dumps(ml["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"], indent=2),
        "```",
        "",
        "## Recommendation",
        "",
        recommendation_md,
        "",
        "## Artifacts",
        "",
        f"- Root: `{out_root}`",
        f"- `strict/`, `lenient/`: audit, summaries, retained/failed draws, `metrics.json`, `plots/`.",
        f"- `comparison_plots/`: side-by-side PIT, coverage, retention, failure comparison.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(report_path, out_root / "heldout_decoder_mode_comparison_report.md")

    print("Done.", out_root, flush=True)
    print("Eligible events:", len(events_eval), flush=True)
    print("Strict mean retained fraction:", ms["mean_retained_fraction"], flush=True)
    print("Lenient mean retained fraction:", ml["mean_retained_fraction"], flush=True)


if __name__ == "__main__":
    main()
