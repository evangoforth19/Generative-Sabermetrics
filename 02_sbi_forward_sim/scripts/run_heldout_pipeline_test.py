#!/usr/bin/env python3
"""
End-to-end held-out test pipeline: p(u|g,p) x p(z|u,g,p) x physics decoder -> (EV, LA, SA).

No training. Frozen stage-u and stage-z checkpoints only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.data_u import (  # noqa: E402
    load_all_u_data,
    load_all_u_data_player_support,
    load_standardization_stats,
    merge_player_constants,
)
from sbi_forward_sim.src.data_z import load_all_z_data_native_circular, load_all_z_data_native_trunc_ex  # noqa: E402
from sbi_forward_sim.src.heldout_forward_inference import (  # noqa: E402
    build_corrected_eligibility_audit,
    merge_baseline_master_row,
    u_encoder_inputs_from_row,
    z_base_x_num_from_row,
    z_categoricals_from_row,
)
from sbi_forward_sim.src.feature_contract_z import load_feature_contract_z  # noqa: E402
from sbi_forward_sim.src.models_u import SharedMixtureGaussianVonMisesUNet  # noqa: E402
from sbi_forward_sim.src.models_z import (  # noqa: E402
    ConditionalHybridGaussVonMisesMixtureZ,
    ConditionalHybridGaussVonMisesTruncExZ,
)
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
    sample_z_hybrid_trunc_ex_batched,
)
from sbi_forward_sim.src.target_transforms_u import PlayerSpecificBoundedLogitZScoreTransform  # noqa: E402
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


def _build_z_model(
    cfg: dict,
    vocabs: dict,
    device: torch.device,
    *,
    feature_contract: dict[str, Any] | None = None,
) -> ConditionalHybridGaussVonMisesMixtureZ:
    mcfg = cfg["model"]
    fc = feature_contract or load_feature_contract_z(cfg, Path(__file__).resolve().parents[1])
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


def _build_z_model_trunc_ex(
    cfg: dict,
    vocabs: dict,
    device: torch.device,
    *,
    feature_contract: dict[str, Any] | None = None,
) -> ConditionalHybridGaussVonMisesTruncExZ:
    mcfg = cfg["model"]
    fc = feature_contract or load_feature_contract_z(cfg, Path(__file__).resolve().parents[1])
    num_f = len(fc["x_numeric_zscore_column_order"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesTruncExZ(
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
        sigma_floor=float(mcfg.get("sigma_floor", 1e-3)),
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




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-run-dir", type=Path, required=True)
    ap.add_argument("--z-run-dir", type=Path, required=True)
    ap.add_argument(
        "--n-samples",
        type=int,
        default=1200,
        help="Fixed mode: attempted joint samples per event. Adaptive mode (--target-admissible-per-event): batch size per sampling round.",
    )
    ap.add_argument(
        "--target-admissible-per-event",
        type=int,
        default=None,
        help="If set, keep sampling in batches of --n-samples until this many admissible decoder draws per event (or --max-attempts-per-event).",
    )
    ap.add_argument(
        "--max-attempts-per-event",
        type=int,
        default=500_000,
        help="Hard cap on joint draws per event when using --target-admissible-per-event.",
    )
    ap.add_argument("--seed", type=int, default=20260408)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--decode-chunk-size",
        type=int,
        default=2500,
        help="Decode and score this many joint draws per decoder batch (memory control).",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="If set, only evaluate the first N eligible events (sorted by event_id). For smoke tests.",
    )
    ap.add_argument(
        "--no-event-progress",
        action="store_true",
        help="With --target-admissible-per-event, suppress one stdout line per finished event.",
    )
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="Write eligibility audit tables under output dir and exit (no sampling).",
    )
    ap.add_argument(
        "--report-filename",
        type=str,
        default="heldout_pipeline_test_corrected_event_universe_report.md",
        help="Markdown report name under reports/ (and copied to output dir).",
    )
    ap.add_argument(
        "--pilot-metrics-json",
        type=Path,
        default=None,
        help="Optional metrics.json from a prior run for comparison in large-scale reports.",
    )
    ap.add_argument(
        "--u-d-tilde-limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="If set, reject stage-u d_tilde samples outside this degree interval before stage-z.",
    )
    ap.add_argument(
        "--decoded-sa-limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="If set, discard decoded admissible draws whose final SA is outside this degree interval.",
    )
    ap.add_argument(
        "--u-t-ang-multiplier",
        type=float,
        default=1.0,
        help="Inference-only multiplier applied to saved stage-u angular temperature T_ang.",
    )
    ap.add_argument(
        "--u-t-va-multiplier",
        type=float,
        default=1.0,
        help="Inference-only multiplier applied to saved stage-u VA covariance temperature T_va.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    decode_chunk = max(1, int(args.decode_chunk_size))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = (project_root / "outputs" / "heldout_pipeline_test" / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    u_run = args.u_run_dir.resolve()
    z_run = args.z_run_dir.resolve()
    u_ckpt = _load_ckpt(u_run / "checkpoint.pt")
    z_ckpt = _load_ckpt(z_run / "checkpoint.pt")
    u_cfg = u_ckpt["config"]
    z_cfg = z_ckpt["config"]
    u_vocabs: dict = u_ckpt["vocabs"]
    z_vocabs: dict = z_ckpt["vocabs"]

    z_frozen_path = z_run / "feature_contract_frozen.json"
    if z_frozen_path.is_file():
        z_fc = json.loads(z_frozen_path.read_text(encoding="utf-8"))
    else:
        z_fc = load_feature_contract_z(z_cfg, project_root)
    z_num_order = list(z_fc["x_numeric_zscore_column_order"])
    master_path = project_root / u_cfg["paths"]["context_master"]
    master = pd.read_parquet(master_path)
    baseline_path = project_root / "data_processed" / "baseline_direct_y_test.parquet"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Need observed y table: {baseline_path}")
    baseline = pd.read_parquet(baseline_path)
    if "split" in baseline.columns:
        baseline = baseline.loc[baseline["split"].eq("test")].copy()

    stats_u = _prepare_stats_u(project_root, u_cfg, master)
    stats = _prepare_stats_z(project_root, z_cfg, z_fc, master)
    u_cols_tpl = ("v_ss_tilde", "a_tilde", "d_tilde")
    audit_df, audit_summary = build_corrected_eligibility_audit(
        baseline, master, stats_u, stats, u_cfg, z_num_order, u_vocabs, z_vocabs, u_cols_tpl
    )
    audit_df.to_csv(out_dir / "event_eligibility_audit.csv", index=False)
    audit_summary["generated_utc"] = stamp
    (out_dir / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")

    if args.audit_only:
        print("Audit-only: wrote", out_dir / "event_eligibility_audit.csv")
        print(json.dumps(audit_summary, indent=2))
        return

    u_head = str(u_ckpt.get("head_family", ""))
    u_bounded_support = u_head == "shared_gaussian_vm_d_bounded_support"
    u_va_transform = None
    if u_bounded_support:
        u_bundle = load_all_u_data_player_support(u_cfg, project_root, vocabs_override=u_vocabs)
        train_u = u_bundle["train"]
        if "target_transform" in u_ckpt:
            u_va_transform = PlayerSpecificBoundedLogitZScoreTransform.from_state_dict(
                u_ckpt["target_transform"]
            )
    else:
        train_u, _, _, _ = load_all_u_data(u_cfg, project_root, vocabs_override=u_vocabs)
    z_trunc_ex = str(z_ckpt.get("model_family", "")) == "hybrid_gauss_vonmises_trunc_ex"
    if z_trunc_ex:
        train_z, _, _, _ = load_all_z_data_native_trunc_ex(
            z_cfg, project_root, vocabs_override=z_vocabs, feature_contract=z_fc
        )
    else:
        train_z, _, _, _ = load_all_z_data_native_circular(
            z_cfg, project_root, vocabs_override=z_vocabs, feature_contract=z_fc
        )

    events_eval = sorted(int(x) for x in audit_df.loc[audit_df["eligible_for_pipeline"], "event_id"].tolist())
    if not events_eval:
        raise RuntimeError(
            "No eligible held-out events (see event_eligibility_audit.csv). "
            "Check baseline targets, decoder master row, and stage-u / stage-z encodings."
        )
    if args.max_events is not None:
        cap = int(args.max_events)
        if cap < 1:
            raise ValueError("--max-events must be >= 1")
        events_eval = events_eval[:cap]
    # Decoder-required columns on master (fail loud)
    need_m = list(DECODER_G_NUMERIC_REQUIRED) + ["event_id", "batter_name"]
    miss_m = [c for c in need_m if c not in master.columns]
    if miss_m:
        raise DecoderInputError(f"sbi_context_event_master.parquet missing columns: {miss_m}")
    spin_ok = (
        "spin_axis_deg" in master.columns
        or ("spin_axis_sin" in master.columns and "spin_axis_cos" in master.columns)
    )
    if not spin_ok:
        raise DecoderInputError("master missing spin axis bundle")

    p_cols = list(DECODER_P_REQUIRED)
    miss_p = [c for c in p_cols if c not in master.columns]
    if miss_p:
        raise DecoderInputError(f"context master missing p columns for decoder: {miss_p}")

    # Models
    u_model = _build_u_model(u_cfg, u_vocabs, device)
    u_model.load_state_dict(u_ckpt["model_state"])
    u_model.eval()
    if z_trunc_ex:
        z_model = _build_z_model_trunc_ex(z_cfg, z_vocabs, device, feature_contract=z_fc)
    else:
        z_model = _build_z_model(z_cfg, z_vocabs, device, feature_contract=z_fc)
    z_model.load_state_dict(z_ckpt["model_state"])
    z_model.eval()

    T_va_base = float(u_ckpt["temperature_T_va"])
    T_ang_u_base = float(u_ckpt["temperature_T_ang"])
    T_va = T_va_base * float(args.u_t_va_multiplier)
    T_ang_u = T_ang_u_base * float(args.u_t_ang_multiplier)
    kappa_max_u = float(u_ckpt.get("kappa_max", u_cfg["model"].get("kappa_max", 120.0)))

    T_gauss_z = float(z_ckpt["temperature_T_gauss"])
    T_ang_z = float(z_ckpt["temperature_T_ang"])
    T_gx = z_ckpt.get("temperature_T_gauss_x")
    T_gy = z_ckpt.get("temperature_T_gauss_y")
    if T_gx is not None and T_gy is not None:
        T_gx_f = float(T_gx)
        T_gy_f = float(T_gy)
    else:
        T_gx_f = None
        T_gy_f = None

    tm_u = torch.tensor(train_u.target_means, dtype=torch.float32, device=device)
    ts_u = torch.tensor(train_u.target_stds, dtype=torch.float32, device=device)
    gm_z = torch.tensor(train_z.gauss_means, dtype=torch.float32, device=device)
    gs_z = torch.tensor(train_z.gauss_stds, dtype=torch.float32, device=device)

    batch_size = int(args.n_samples)
    target_adm = args.target_admissible_per_event
    max_attempts_ev = int(args.max_attempts_per_event)
    show_event_progress = target_adm is not None and not args.no_event_progress
    rows_draw: list[dict[str, Any]] = []
    rows_fail: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []

    g_gen_u = torch.Generator(device=device)
    g_gen_u.manual_seed(args.seed)
    g_gen_z = torch.Generator(device=device)
    g_gen_z.manual_seed(args.seed + 17)

    u_cols = u_cols_tpl
    u_d_tilde_limits = (
        (float(args.u_d_tilde_limits[0]), float(args.u_d_tilde_limits[1]))
        if args.u_d_tilde_limits is not None
        else None
    )
    decoded_sa_limits = (
        (float(args.decoded_sa_limits[0]), float(args.decoded_sa_limits[1]))
        if args.decoded_sa_limits is not None
        else None
    )

    def decode_one_round(
        eid: int,
        batter: str,
        g_dec: dict[str, Any],
        p_dec: dict[str, Any],
        n_this: int,
        draw_base: int,
        u_np: np.ndarray,
        x_samp: np.ndarray,
        ey_samp: np.ndarray,
        psi_samp: np.ndarray,
        th_samp: np.ndarray | None,
        fails_by_reason: dict[str, int],
        adm_sink: list[dict[str, Any]] | None,
        adm_cap: int | None,
        *,
        ex_samp: np.ndarray | None = None,
    ) -> tuple[int, int]:
        """Decode n_this joint draws starting at draw_base global indices.

        adm_sink: if not None, append admissible row dicts here until len reaches adm_cap (if set).
        Returns (n_admissible_recorded, n_admissible_seen_in_round).
        """
        inps: list[DecoderInputs] = []
        u_rows: list[dict[str, float]] = []
        z_rows: list[dict[str, float]] = []
        for j in range(n_this):
            uj = {
                "v_ss_tilde": float(u_np[j, 0]),
                "a_tilde": float(u_np[j, 1]),
                "d_tilde": float(u_np[j, 2]),
            }
            if ex_samp is not None:
                zj = {
                    "x": float(x_samp[j]),
                    "e_y_star": float(ey_samp[j]),
                    "psi_deg": float(psi_samp[j]),
                    "e_x": float(ex_samp[j]),
                }
            else:
                assert th_samp is not None
                zj = {
                    "x": float(x_samp[j]),
                    "e_y_star": float(ey_samp[j]),
                    "psi_deg": float(psi_samp[j]),
                    "theta_deg": float(th_samp[j]),
                }
            u_rows.append(uj)
            z_rows.append(zj)
            gj = draw_base + j
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

        n_seen = 0
        n_recorded = 0
        for j0 in range(0, n_this, decode_chunk):
            j1 = min(j0 + decode_chunk, n_this)
            chunk_inps = inps[j0:j1]
            try:
                df_batch = decode_bip_batch(chunk_inps)
            except DecoderInputError as ex:
                for j in range(j0, j1):
                    uj, zj = u_rows[j], z_rows[j]
                    rows_fail.append(
                        {
                            "event_id": eid,
                            "draw_idx": draw_base + j,
                            "failure_reason": f"batch_validate:{ex}",
                            **{f"u_{k}": v for k, v in uj.items()},
                            **{f"z_{k}": v for k, v in zj.items()},
                        }
                    )
                fails_by_reason["batch_validate"] = fails_by_reason.get("batch_validate", 0) + (j1 - j0)
            else:
                for rel, j in enumerate(range(j0, j1)):
                    out = df_batch.iloc[rel]
                    uj, zj = u_rows[j], z_rows[j]
                    gj = draw_base + j
                    if not bool(out.get("admissible", False)):
                        reason = str(out.get("failure_reason") or "inadmissible")
                        rows_fail.append(
                            {
                                "event_id": eid,
                                "draw_idx": gj,
                                "failure_reason": reason,
                                **{f"u_{k}": v for k, v in uj.items()},
                                **{f"z_{k}": v for k, v in zj.items()},
                            }
                        )
                        fails_by_reason[reason] = fails_by_reason.get(reason, 0) + 1
                        continue
                    if decoded_sa_limits is not None:
                        sa_val = _row_float(out, "SA")
                        sa_low, sa_high = decoded_sa_limits
                        if not np.isfinite(sa_val) or sa_val < sa_low or sa_val > sa_high:
                            reason = "decoded_sa_outside_limits"
                            rows_fail.append(
                                {
                                    "event_id": eid,
                                    "draw_idx": gj,
                                    "failure_reason": reason,
                                    "decoded_SA": sa_val,
                                    "decoded_sa_limits_low": sa_low,
                                    "decoded_sa_limits_high": sa_high,
                                    **{f"u_{k}": v for k, v in uj.items()},
                                    **{f"z_{k}": v for k, v in zj.items()},
                                }
                            )
                            fails_by_reason[reason] = fails_by_reason.get(reason, 0) + 1
                            continue
                    n_seen += 1
                    row_adm = {
                        "event_id": eid,
                        "batter_name": batter,
                        "draw_idx": gj,
                        "u_v_ss_tilde": uj["v_ss_tilde"],
                        "u_a_tilde": uj["a_tilde"],
                        "u_d_tilde": uj["d_tilde"],
                        "z_x": zj["x"],
                        "z_psi_deg": zj["psi_deg"],
                        "z_e_y_star": zj["e_y_star"],
                        "z_theta_deg": float(zj["theta_deg"]) if "theta_deg" in zj else float("nan"),
                        "z_e_x": float(zj["e_x"]) if "e_x" in zj else float("nan"),
                        "EV": _row_float(out, "EV"),
                        "LA": _row_float(out, "LA"),
                        "SA": _row_float(out, "SA"),
                        "e_x": _row_float(out, "e_x"),
                        "omega_plus_rad_s": _row_float(out, "omega_plus_rad_s"),
                        "V_n_plus_fps": _row_float(out, "V_n_plus_fps"),
                        "V_t_plus_fps": _row_float(out, "V_t_plus_fps"),
                        "admissible": True,
                        "failure_reason": "",
                    }
                    if adm_sink is not None:
                        if adm_cap is None or len(adm_sink) < adm_cap:
                            adm_sink.append(row_adm)
                            n_recorded += 1
                    else:
                        rows_draw.append(row_adm)
                        n_recorded += 1
        return n_recorded, n_seen

    n_events_total = len(events_eval)
    for i_ev, eid in enumerate(events_eval, start=1):
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
        base_z = z_base_x_num_from_row(row, z_num_order, stats, u_cols)
        cat_z_np = z_categoricals_from_row(row, z_vocabs)

        if target_adm is None:
            S = batch_size
            u_pack = sample_u_shared_gaussian_vm(
                u_model,
                xu,
                catu,
                n_samples=S,
                T_va=T_va,
                T_ang=T_ang_u,
                kappa_max=kappa_max_u,
                target_means=tm_u,
                target_stds=ts_u,
                generator=g_gen_u,
                d_tilde_deg_limits=u_d_tilde_limits,
                va_transform=u_va_transform,
                player_ids=np.full(S, int(cat_u_np["batter_name"][0]), dtype=np.int64)
                if u_va_transform is not None
                else None,
            )
            u_np = np.stack(
                [
                    u_pack["v_ss_tilde"].cpu().numpy(),
                    u_pack["a_tilde"].cpu().numpy(),
                    u_pack["d_tilde"].cpu().numpy(),
                ],
                axis=1,
            )
            x_z_np = patch_z_x_num_with_u(base_z, z_num_order, u_cols, u_np, stats)
            x_z = torch.from_numpy(x_z_np).to(device)
            catz = {
                k: torch.full((S,), int(cat_z_np[k][0]), device=device, dtype=torch.long)
                for k in sorted(z_vocabs.keys())
            }
            if z_trunc_ex:
                z_pack = sample_z_hybrid_trunc_ex_batched(
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
                ex_samp = z_pack["e_x"][:, 0].cpu().numpy()
                th_samp = None
            else:
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
                ex_samp = None

            attempted = S
            fails_by_reason: dict[str, int] = {}
            adm_rec, _ = decode_one_round(
                eid,
                batter,
                g_dec,
                p_dec,
                S,
                0,
                u_np,
                x_samp,
                ey_samp,
                psi_samp,
                th_samp,
                fails_by_reason,
                adm_sink=None,
                adm_cap=None,
                ex_samp=ex_samp,
            )
            adm = adm_rec
        else:
            rows_adm_cur: list[dict[str, Any]] = []
            fails_by_reason = {}
            attempted = 0
            draw_base = 0
            while len(rows_adm_cur) < target_adm and attempted < max_attempts_ev:
                n_this = min(batch_size, max_attempts_ev - attempted)
                if n_this <= 0:
                    break
                u_pack = sample_u_shared_gaussian_vm(
                    u_model,
                    xu,
                    catu,
                    n_samples=n_this,
                    T_va=T_va,
                    T_ang=T_ang_u,
                    kappa_max=kappa_max_u,
                    target_means=tm_u,
                    target_stds=ts_u,
                    generator=g_gen_u,
                    d_tilde_deg_limits=u_d_tilde_limits,
                    va_transform=u_va_transform,
                    player_ids=np.full(n_this, int(cat_u_np["batter_name"][0]), dtype=np.int64)
                    if u_va_transform is not None
                    else None,
                )
                u_np = np.stack(
                    [
                        u_pack["v_ss_tilde"].cpu().numpy(),
                        u_pack["a_tilde"].cpu().numpy(),
                        u_pack["d_tilde"].cpu().numpy(),
                    ],
                    axis=1,
                )
                x_z_np = patch_z_x_num_with_u(base_z, z_num_order, u_cols, u_np, stats)
                x_z = torch.from_numpy(x_z_np).to(device)
                catz = {
                    k: torch.full((n_this,), int(cat_z_np[k][0]), device=device, dtype=torch.long)
                    for k in sorted(z_vocabs.keys())
                }
                if z_trunc_ex:
                    z_pack = sample_z_hybrid_trunc_ex_batched(
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
                    ex_samp = z_pack["e_x"][:, 0].cpu().numpy()
                    th_samp = None
                else:
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
                    ex_samp = None
                decode_one_round(
                    eid,
                    batter,
                    g_dec,
                    p_dec,
                    n_this,
                    draw_base,
                    u_np,
                    x_samp,
                    ey_samp,
                    psi_samp,
                    th_samp,
                    fails_by_reason,
                    adm_sink=rows_adm_cur,
                    adm_cap=target_adm,
                    ex_samp=ex_samp,
                )
                attempted += n_this
                draw_base += n_this

            if len(rows_adm_cur) < target_adm:
                raise RuntimeError(
                    f"event_id={eid}: only {len(rows_adm_cur)} admissible draws after "
                    f"{attempted} attempts (target={target_adm}, max_attempts_per_event={max_attempts_ev})."
                )
            rows_draw.extend(rows_adm_cur[:target_adm])
            adm = target_adm

        frac = adm / max(attempted, 1)
        low_adm_flag = bool(frac < 0.05 or adm < 30) if target_adm is None else bool(frac < 0.01)
        event_summaries.append(
            {
                "event_id": eid,
                "batter_name": batter,
                "split": "test",
                "observed_EV": obs_ev,
                "observed_LA": obs_la,
                "observed_SA": obs_sa,
                "attempted_draws": attempted,
                "admissible_draws": adm,
                "admissible_fraction": frac,
                "failure_reason_counts_json": json.dumps(fails_by_reason),
                "flag_low_admissible": low_adm_flag,
            }
        )
        if show_event_progress:
            print(
                f"[heldout_progress] event {i_ev}/{n_events_total} event_id={eid} "
                f"admissible={adm} attempted={attempted}",
                flush=True,
            )

    df_draw = pd.DataFrame(rows_draw)
    df_fail = pd.DataFrame(rows_fail)
    df_ev = pd.DataFrame(event_summaries)

    # Fill predictive summaries per event from admissible draws only
    def qdict(s: pd.Series, qs: list[float]) -> dict[str, float]:
        s = s.dropna().to_numpy()
        if len(s) == 0:
            return {f"q{int(q*100)}": float("nan") for q in qs}
        out = {}
        for q in qs:
            out[f"q{int(q*100)}"] = float(np.quantile(s, q))
        return out

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
            qd = qdict(sub, qlist)
            for k in qd:
                qcols[f"pred_{tgt}_{k}"].append(qd[k])
        df_ev[f"pred_{tgt}_mean"] = means
        df_ev[f"pred_{tgt}_median"] = medians
        df_ev[f"pred_{tgt}_std"] = stds
        for k, v in qcols.items():
            df_ev[k] = v

    sampling_mode = "adaptive_until_admissible" if target_adm is not None else "fixed_attempts"
    if target_adm is None:
        attempts_per_event_report = str(batch_size)
    else:
        attempts_per_event_report = (
            f"adaptive: batch_size={batch_size} until admissible={target_adm} "
            f"(max_attempts_per_event={max_attempts_ev})"
        )

    # Global metrics (event-level on mean prediction + sample-based)
    metrics: dict[str, Any] = {
        "n_events": int(len(events_eval)),
        "n_samples_batch_per_event": batch_size,
        "n_samples_requested_per_event": batch_size,
        "sampling_mode": sampling_mode,
        "target_admissible_per_event": target_adm,
        "max_attempts_per_event_cap": max_attempts_ev if target_adm is not None else None,
        "total_attempted": int(df_ev["attempted_draws"].sum()),
        "total_admissible": int(df_ev["admissible_draws"].sum()),
        "mean_admissible_fraction": float(df_ev["admissible_fraction"].mean()),
        "frozen_stage_u_run": str(u_run),
        "frozen_stage_z_run": str(z_run),
        "observed_y_source": str(baseline_path),
        "master_path": str(master_path),
        "held_out_event_universe": "baseline_direct_y_test_official_split; stage_u/stage_z inputs built from baseline+master; u_test/z_test parquets not required",
        "audit_eligible_count": int(audit_summary["n_eligible_for_forward_pipeline"]),
        "note_LA": "LA aligns with decoder launch_angle_deg (theta_star lineage in reduced model); linear error in degrees is used; |LA| typically < 90 so wrap rarely differs.",
        "note_SA": "SA is spray angle deg; wrapped error used where noted.",
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
    metrics["stage_u_inference_temperature"] = {
        "T_va_base": T_va_base,
        "T_va_multiplier": float(args.u_t_va_multiplier),
        "T_va_used": T_va,
        "T_ang_base": T_ang_u_base,
        "T_ang_multiplier": float(args.u_t_ang_multiplier),
        "T_ang_used": T_ang_u,
    }
    metrics["support_policy"] = {
        "u_d_tilde_limits": list(u_d_tilde_limits) if u_d_tilde_limits is not None else None,
        "decoded_sa_limits": list(decoded_sa_limits) if decoded_sa_limits is not None else None,
    }

    def _tail_block(vals: pd.Series | np.ndarray, thresholds: tuple[float, ...]) -> dict[str, Any]:
        arr = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "n": 0,
                "mean": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "p99_9": float("nan"),
                "max": float("nan"),
                "threshold_counts": {str(t): 0 for t in thresholds},
                "threshold_rates": {str(t): float("nan") for t in thresholds},
            }
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "p99_9": float(np.quantile(arr, 0.999)),
            "max": float(np.max(arr)),
            "threshold_counts": {str(t): int(np.sum(arr > t)) for t in thresholds},
            "threshold_rates": {str(t): float(np.mean(arr > t)) for t in thresholds},
        }

    tail_thresholds = {
        "EV": (115.0, 120.0, 125.0),
        "u_v_ss_tilde": (90.0, 95.0, 100.0),
    }
    metrics["tail_diagnostics"] = {
        "global": {
            "EV": _tail_block(df_draw["EV"], tail_thresholds["EV"]),
            "u_v_ss_tilde": _tail_block(df_draw["u_v_ss_tilde"], tail_thresholds["u_v_ss_tilde"]),
            "SA": _tail_block(df_draw["SA"], (45.0,)),
        },
        "per_hitter_csv": "tail_diagnostics_by_hitter.csv",
    }

    hitter_tail_rows: list[dict[str, Any]] = []
    for hitter, hdf in df_draw.groupby("batter_name", sort=True):
        row_tail: dict[str, Any] = {"batter_name": str(hitter), "n_draws": int(len(hdf))}
        for col, thresholds in tail_thresholds.items():
            block = _tail_block(hdf[col], thresholds)
            for key in ("mean", "p95", "p99", "p99_9", "max"):
                row_tail[f"{col}_{key}"] = block[key]
            for thr, val in block["threshold_rates"].items():
                row_tail[f"{col}_rate_gt_{thr}"] = val
        sa_block = _tail_block(hdf["SA"], (45.0,))
        row_tail["SA_min"] = float(pd.to_numeric(hdf["SA"], errors="coerce").min())
        row_tail["SA_max"] = sa_block["max"]
        row_tail["SA_rate_gt_45"] = sa_block["threshold_rates"]["45.0"]
        row_tail["SA_rate_lt_neg45"] = float(np.mean(pd.to_numeric(hdf["SA"], errors="coerce") < -45.0))
        hitter_tail_rows.append(row_tail)
    pd.DataFrame(hitter_tail_rows).to_csv(out_dir / "tail_diagnostics_by_hitter.csv", index=False)

    # Wrapped MAE/RMSE using posterior predictive mean vs obs (admissible draws)
    w_mae_sa = []
    w_sq_sa = []
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

    # --- save tables ---
    df_ev.to_parquet(out_dir / "predictive_summary_by_event.parquet", index=False)
    df_ev.to_csv(out_dir / "predictive_summary_by_event.csv", index=False)
    df_draw.to_parquet(out_dir / "predictive_draws_admissible.parquet", index=False)
    df_draw.to_csv(out_dir / "predictive_draws_admissible.csv", index=False)
    if len(df_fail):
        df_fail.to_parquet(out_dir / "predictive_draws_failed.parquet", index=False)
        df_fail.to_csv(out_dir / "predictive_draws_failed.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    sum_rows: list[list[Any]] = [
        ["n_events", metrics["n_events"]],
        ["n_samples_requested_per_event", metrics["n_samples_requested_per_event"]],
        ["total_attempted", metrics["total_attempted"]],
        ["total_admissible", metrics["total_admissible"]],
        ["mean_admissible_fraction", metrics["mean_admissible_fraction"]],
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
    pd.DataFrame(sum_rows, columns=["metric", "value"]).to_csv(out_dir / "metrics_summary.csv", index=False)
    tail = metrics["tail_diagnostics"]["global"]
    tail_rows: list[list[Any]] = []
    for name, block in tail.items():
        for key in ("mean", "p95", "p99", "p99_9", "max"):
            tail_rows.append([f"{name}_{key}", block[key]])
        for thr, val in block["threshold_rates"].items():
            tail_rows.append([f"{name}_rate_gt_{thr}", val])
    pd.DataFrame(tail_rows, columns=["metric", "value"]).to_csv(out_dir / "tail_diagnostics_summary.csv", index=False)

    # --- plots ---
    def _pit_plot(vals: list[float], name: str, title: str) -> None:
        v = [x for x in vals if np.isfinite(x)]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(v, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("PIT")
        fig.tight_layout()
        fig.savefig(plots_dir / f"pit_{name}.png", dpi=120)
        plt.close(fig)

    _pit_plot(pits_ev, "EV", "PIT EV (predictive samples)")
    _pit_plot(pits_la, "LA", "PIT LA")
    _pit_plot(pits_sa, "SA", "PIT SA")

    for tgt, obs_col in [("EV", "observed_EV"), ("LA", "observed_LA"), ("SA", "observed_SA")]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(df_ev[obs_col], df_ev[f"pred_{tgt}_mean"], alpha=0.8, edgecolors="k", linewidths=0.5)
        mx = float(np.nanmax([df_ev[obs_col].max(), df_ev[f"pred_{tgt}_mean"].max()]))
        mn = float(np.nanmin([df_ev[obs_col].min(), df_ev[f"pred_{tgt}_mean"].min()]))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel(f"Observed {tgt}")
        ax.set_ylabel(f"Pred mean {tgt}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"scatter_mean_vs_obs_{tgt}.png", dpi=120)
        plt.close(fig)

    for tgt, obs_col in [("EV", "observed_EV"), ("LA", "observed_LA"), ("SA", "observed_SA")]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(df_ev[obs_col], df_ev[f"pred_{tgt}_median"], alpha=0.8, edgecolors="k", linewidths=0.5)
        mx = float(np.nanmax([df_ev[obs_col].max(), df_ev[f"pred_{tgt}_median"].max()]))
        mn = float(np.nanmin([df_ev[obs_col].min(), df_ev[f"pred_{tgt}_median"].min()]))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel(f"Observed {tgt}")
        ax.set_ylabel(f"Pred median {tgt}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"scatter_median_vs_obs_{tgt}.png", dpi=120)
        plt.close(fig)

    for tgt, obs_col in [("EV", "observed_EV"), ("LA", "observed_LA"), ("SA", "observed_SA")]:
        fig, ax = plt.subplots(figsize=(4, 4))
        res = df_ev[f"pred_{tgt}_mean"].to_numpy() - df_ev[obs_col].to_numpy()
        ax.scatter(df_ev[obs_col], res, alpha=0.8, edgecolors="k", linewidths=0.5)
        ax.axhline(0.0, color="r", ls="--", lw=1)
        ax.set_xlabel(f"Observed {tgt}")
        ax.set_ylabel(f"Residual (mean pred - obs)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"residual_{tgt}.png", dpi=120)
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
        ax.set_title(f"{tgt} central interval coverage (n={len(df_ev)} events)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"coverage_intervals_{tgt}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df_ev["admissible_fraction"].to_numpy(), bins=min(20, max(5, len(df_ev))), color="coral", edgecolor="black")
    ax.set_xlabel("Admissible fraction (per event)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "hist_admissible_fraction.png", dpi=120)
    plt.close(fig)

    if len(df_fail) and "failure_reason" in df_fail.columns:
        vc = df_fail["failure_reason"].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(6, 4))
        vc.plot(kind="bar", ax=ax, color="gray")
        ax.set_title("Decoder failure reasons (top 20)")
        fig.tight_layout()
        fig.savefig(plots_dir / "failure_reason_counts.png", dpi=120)
        plt.close(fig)

    # Example predictive distributions (up to 3 events with most admissible draws)
    top_e = df_ev.nlargest(3, "admissible_draws")["event_id"].tolist()
    for eid in top_e:
        sub = df_draw.loc[df_draw["event_id"] == eid]
        er = df_ev.loc[df_ev["event_id"] == eid].iloc[0]
        if len(sub) < 5:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, tgt, ox in zip(axes, ("EV", "LA", "SA"), ("observed_EV", "observed_LA", "observed_SA"), strict=True):
            ax.hist(sub[tgt].to_numpy(), bins=30, color="skyblue", edgecolor="black", alpha=0.85)
            ax.axvline(er[ox], color="red", lw=2, label="observed")
            ax.set_title(tgt)
            ax.legend(fontsize=8)
        fig.suptitle(f"Event {eid} predictive marginals (admissible draws)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"pred_dist_event_{eid}.png", dpi=120)
        plt.close(fig)

    manifest = {
        "timestamp_utc": stamp,
        "git_hash": _git_hash(),
        "stage_u_checkpoint": str(u_run / "checkpoint.pt"),
        "stage_z_checkpoint": str(z_run / "checkpoint.pt"),
        "n_events_evaluated": len(events_eval),
        "event_ids": events_eval,
        "n_samples_batch_per_event": batch_size,
        "n_samples_per_event_requested": batch_size,
        "sampling_mode": sampling_mode,
        "target_admissible_per_event": target_adm,
        "max_attempts_per_event_cap": max_attempts_ev if target_adm is not None else None,
        "decode_chunk_size": decode_chunk,
        "support_policy": metrics["support_policy"],
        "stage_u_inference_temperature": metrics["stage_u_inference_temperature"],
        "total_attempted": metrics["total_attempted"],
        "total_admissible": metrics["total_admissible"],
        "outputs_dir": str(out_dir),
        "audit_summary_path": "audit_summary.json",
        "event_eligibility_audit_path": "event_eligibility_audit.csv",
        "report_filename": args.report_filename,
        "held_out_event_universe": "baseline_test_baseline_plus_master_encoders_no_u_z_test_rows",
        "artifacts": {
            "event_eligibility_audit": "event_eligibility_audit.csv",
            "audit_summary": "audit_summary.json",
            "predictive_summary_by_event": "predictive_summary_by_event.parquet",
            "predictive_draws_admissible": "predictive_draws_admissible.parquet",
            "predictive_draws_failed": "predictive_draws_failed.parquet" if len(df_fail) else None,
            "metrics": "metrics.json",
            "metrics_summary_csv": "metrics_summary.csv",
            "plots_directory": "plots/",
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Report
    report_path = project_root / "reports" / args.report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cov = metrics["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]
    crps = metrics["probabilistic_sample_based"]["mean_CRPS"]
    pacc = metrics["point_accuracy_event_mean_prediction"]

    pilot_path = args.pilot_metrics_json
    if pilot_path is None:
        if "corrected" in args.report_filename:
            _guess = project_root / "outputs" / "heldout_pipeline_test" / "20260407_205501Z" / "metrics.json"
            if _guess.is_file():
                pilot_path = _guess
        elif "large_scale" in args.report_filename:
            _guess = project_root / "outputs" / "heldout_pipeline_test" / "20260407_204536Z" / "metrics.json"
            if _guess.is_file():
                pilot_path = _guess
    pilot_metrics: dict[str, Any] | None = None
    if pilot_path is not None and Path(pilot_path).is_file():
        pilot_metrics = json.loads(Path(pilot_path).read_text(encoding="utf-8"))

    if "corrected" in args.report_filename:
        title = "# Held-out full pipeline — corrected event universe"
    elif "large_scale" in args.report_filename:
        title = "# Held-out full pipeline test — large-scale evaluation"
    else:
        title = "# Held-out full pipeline test (generative → decoder → EV, LA, SA)"
    lines = [
        title,
        "",
        "## Concept: corrected held-out universe",
        "",
        "**Previous (incorrect) gate:** pipeline required an event to appear in **`u_test.parquet`**, **`z_test.parquet`**, "
        "and **`baseline_direct_y_test`**. Because **`z_test`** only contained **7** unique `event_id`, the benchmark "
        "collapsed to 7 events even though the official test split has **~1010** baseline rows.",
        "",
        "**Corrected gate (this run):** eligible events are **official-holdout baseline / direct-y test rows** that have "
        "**observed (EV, LA, SA)**, **decoder-complete** `sbi_context_event_master` context, and **encodable** stage-u and stage-z "
        "conditioning built from the same fields used in training (categoricals map through frozen vocabs, including `<UNK>`).",
        "No **`z_test` / `u_test` parquet membership** is required for inference.",
        "",
        "## Held-out event-count audit",
        "",
        f"```json\n{json.dumps(audit_summary, indent=2)}\n```",
        "",
        f"- Per-event flags: **`event_eligibility_audit.csv`** in `{out_dir}`.",
        "",
        "## Frozen modules",
        "",
        f"- **Stage-u run:** `{u_run}`",
        f"- **Stage-z run:** `{z_run}`",
        "",
        "## Data sources",
        "",
        f"- Eligible events = **`eligible_for_pipeline`** in `event_eligibility_audit.csv` (baseline test split ∩ encoder ∩ decoder).",
        f"- Decoder **g** from `{master_path.name}` (decoder-complete block, not neural-only `g`).",
        f"- Observed **EV, LA, SA** from `{baseline_path.name}`.",
        "",
        "## Scale",
        "",
        f"- **Events evaluated:** {len(events_eval)}",
        f"- **Draws per event:** {attempts_per_event_report}",
        f"- **Decode chunk size:** {decode_chunk}",
        f"- **Total attempted:** {metrics['total_attempted']}",
        f"- **Total admissible:** {metrics['total_admissible']}",
        f"- **Mean admissible fraction:** {metrics['mean_admissible_fraction']:.4f}",
        "",
        "## Metrics summary",
        "",
        "### Point accuracy (mean predictive vs observed, per-event then averaged)",
        "",
        f"```json\n{json.dumps(pacc, indent=2)}\n```",
        "",
        "### Sample-based CRPS / coverage (central intervals from admissible samples)",
        "",
        f"- Mean CRPS: `{json.dumps(crps)}`",
        f"- Coverage rates: `{json.dumps(cov)}`",
        "",
        "### Wrapped SA (mean prediction vs observed)",
        "",
        f"```json\n{json.dumps(metrics.get('point_accuracy_wrapped_SA_mean', {}), indent=2)}\n```",
        "",
        "## Artifacts",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Plots: `{out_dir / 'plots'}` (PIT, mean & **median** vs obs scatter, residuals, **coverage intervals**, admissibility hist, example marginals).",
        f"- Event-level table schema: identifiers, observed truths, `pred_*` moments & quantiles, admissibility counts.",
        "- Draw-level table: one row per **admissible** predictive draw with full u, z, decoder diagnostics.",
        "",
        "## Baseline comparison note",
        "",
        "Future black-box models should use the **same** `event_id` list, the same observed columns from "
        "`predictive_summary_by_event.csv`, and the same metric definitions (MAE/RMSE/bias, CRPS, interval coverage).",
        "",
        "## Caveats",
        "",
        "- **θ provenance:** stage-z `theta_deg` on training draws uses observed-launch proxy when `theta_star_deg` absent; "
        "pipeline samples θ from the learned conditional the same as stage-z evaluation.",
        "- **Decoder g ⊃ neural g:** kinematics and `release_pos_y` come from context master, not the smaller neural `G_COLUMNS`.",
        "- **Inadmissibility:** rates are reported explicitly; do not compare only admissible subsamples without disclosure.",
        "",
        "### Interpretation",
        "",
        "If coverage is near nominal and CRPS is reasonable relative to marginal spread, the generative stack is "
        "**plausible for baseline benchmarking** on this held-out slice. Low admissible counts per event inflate "
        "uncertainty in sample-based scores.",
        "",
    ]
    if pilot_metrics is not None:
        comp_note = (
            "**Wrong-intersection benchmark:** that file used **7** events (old `u_test ∩ z_test ∩ baseline` logic). "
            "Metrics are **not comparable** event-for-event to this corrected run unless you filter to the same IDs; "
            "the corrected benchmark is the proper **~1010-event** test slice for black-box comparison."
            if pilot_metrics.get("n_events", 0) <= 10 and metrics["n_events"] > 50
            else "Compare attempted/admissible totals and per-target scores; ensure the same `n_samples` and RNG seed when isolating Monte Carlo variance."
        )
        lines.extend(
            [
                "## Comparison to previous held-out run (optional)",
                "",
                f"- Reference metrics file: `{pilot_path}`",
                f"- Reference: attempted {pilot_metrics.get('total_attempted')}, admissible {pilot_metrics.get('total_admissible')}, "
                f"n_events {pilot_metrics.get('n_events')}.",
                f"- This run: attempted {metrics['total_attempted']}, admissible {metrics['total_admissible']}, n_events {metrics['n_events']}.",
                "",
                comp_note,
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(report_path, out_dir / args.report_filename)

    print("Done.", out_dir)
    print("Events:", len(events_eval), "Admissible total:", metrics["total_admissible"])


if __name__ == "__main__":
    main()
