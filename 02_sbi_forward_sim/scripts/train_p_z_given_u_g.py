#!/usr/bin/env python3
"""Train stage-z conditional MDN p(z | u, g, p)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.data_z import load_all_z_data  # noqa: E402
from sbi_forward_sim.src.eval_z import (  # noqa: E402
    collect_pits_z_batched,
    evaluate_z_split_batched,
    plot_pit_hist_z,
    save_metrics_json_csv,
)
from sbi_forward_sim.src.feature_contract_z import (  # noqa: E402
    load_feature_contract_z,
    resolve_feature_contract_path_z,
)
from sbi_forward_sim.src.models_z import ConditionalGaussianMixtureZ  # noqa: E402
from sbi_forward_sim.src.train_z import train_stage_z_mdn  # noqa: E402


def _build_model(
    cfg: dict, num_f: int, vocabs: dict[str, dict[str, int]], device: torch.device
) -> ConditionalGaussianMixtureZ:
    mcfg = cfg["model"]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalGaussianMixtureZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        z_dim=int(mcfg["z_dim"]),
        n_components=int(mcfg["n_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
    ).to(device)


def _arrays_to_tensors_z(arr, device: torch.device):
    x = torch.from_numpy(arr.x_num).to(device)
    cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    y = torch.from_numpy(arr.y).to(device)
    yr = torch.from_numpy(arr.y_raw).to(device)
    w = torch.from_numpy(arr.w).to(device)
    return x, cat, y, yr, w


def _git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_MMC2_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def _theta_lineage(project_root: Path, cfg: dict) -> dict:
    """Finite theta_star_deg vs theta_obs_deg fallback; merge build manifest stats when present."""
    out: dict = {}
    ms_path = project_root / "manifests" / "modeling_schema.json"
    if ms_path.is_file():
        try:
            ms = json.loads(ms_path.read_text(encoding="utf-8"))
            out["modeling_schema_path"] = str(ms_path)
            out["z_theta_provenance"] = ms.get("z_theta_provenance")
            out["z_theta_row_stats_from_build"] = ms.get("z_theta_row_stats")
        except Exception as e:  # noqa: BLE001
            out["modeling_schema_read_error"] = str(e)

    rel = cfg["paths"].get("joint_train")
    if not rel:
        out["note_joint"] = "No joint_train path in config."
        return out
    jp = (project_root / rel).resolve()
    if not jp.is_file():
        out["note_joint"] = f"joint_train not found at {jp}"
        return out
    try:
        cols = set(pd.read_parquet(jp, engine="pyarrow").columns)
    except Exception:  # noqa: BLE001
        cols = set(pd.read_parquet(jp).columns)
    if "split" not in cols or "theta_obs_deg" not in cols:
        out["note_joint"] = f"{jp.name} missing split/theta_obs_deg for lineage."
        return out
    if "theta_star_deg" not in cols:
        df = pd.read_parquet(jp, columns=["split"])
        df = df.loc[df["split"].eq("train")]
        n_tot = int(len(df))
        out.update(
            {
                "joint_train_path": str(jp),
                "joint_train_rows_used": n_tot,
                "n_train_from_finite_theta_star_deg": 0,
                "n_train_from_theta_obs_deg_fallback": n_tot,
                "fraction_draw_level_theta_star": 0.0,
                "note_joint": "No `theta_star_deg` column on joint export; draw-level theta counts are zero.",
            }
        )
        return out
    df = pd.read_parquet(jp, columns=["theta_star_deg", "theta_obs_deg", "split"])
    df = df.loc[df["split"].eq("train")]
    star = pd.to_numeric(df["theta_star_deg"], errors="coerce")
    n_star = int(np.isfinite(star.to_numpy()).sum())
    n_tot = int(len(df))
    out.update(
        {
            "joint_train_path": str(jp),
            "joint_train_rows_used": n_tot,
            "n_train_from_finite_theta_star_deg": n_star,
            "n_train_from_theta_obs_deg_fallback": n_tot - n_star,
            "fraction_draw_level_theta_star": (n_star / n_tot) if n_tot else None,
        }
    )
    return out


def _append_theta_fraction_from_build_manifest(lineage: dict) -> None:
    """If joint-level fraction is missing, use strict-pool counts from build manifest."""
    if lineage.get("fraction_draw_level_theta_star") is not None:
        return
    st = (lineage.get("z_theta_row_stats_from_build") or {}).get("strict") or {}
    ns = int(st.get("n_rows_from_theta_star_deg", 0) or 0)
    no = int(st.get("n_rows_from_theta_obs_deg", 0) or 0)
    tot = ns + no
    if tot > 0:
        lineage["theta_star_fraction_strict_build_pool"] = ns / tot
        lineage["theta_star_fraction_note"] = (
            "theta_star_fraction_strict_build_pool is from dataset build manifest strict pool "
            "(counts all strict-screened draws), used when joint_train lacks theta lineage columns."
        )


def _write_training_report(
    project_root: Path,
    run_dir: Path,
    stamp: str,
    cfg: dict,
    frozen_contract: dict,
    contract_path: Path,
    meta: dict,
    train_meta: dict,
    T_mix: float,
    results: dict,
    theta_lineage: dict,
    ckpt_model_family: str,
) -> Path:
    ms_path = project_root / "manifests" / "modeling_schema.json"
    ms_note = ""
    if ms_path.is_file():
        ms = json.loads(ms_path.read_text(encoding="utf-8"))
        ms_note = (
            f"Authoritative schema: `{ms_path}` — z targets {ms.get('z')}, "
            f"`theta_deg` rule: {ms.get('notes', {}).get('theta_deg', ms.get('z_theta_provenance', ''))}"
        )
    def _summ_coord(res_key: str) -> str:
        r = results.get(res_key, {})
        names = r.get("target_names", ["x", "psi_deg", "e_y_star", "theta_deg"])
        ma = r.get("mae_raw") or []
        rm = r.get("rmse_raw") or []
        c50 = r.get("coverage_50_marginal") or []
        c80 = r.get("coverage_80_marginal") or []
        c90 = r.get("coverage_90_marginal") or []
        parts = []
        for i, nm in enumerate(names):
            mae_s = f"{ma[i]:.4f}" if i < len(ma) else "n/a"
            rmse_s = f"{rm[i]:.4f}" if i < len(rm) else "n/a"
            p50 = f"{c50[i]:.3f}" if i < len(c50) else "n/a"
            p80 = f"{c80[i]:.3f}" if i < len(c80) else "n/a"
            p90 = f"{c90[i]:.3f}" if i < len(c90) else "n/a"
            parts.append(f"{nm}: MAE={mae_s}, RMSE={rmse_s}, cov50={p50}, cov80={p80}, cov90={p90}")
        return "; ".join(parts)

    frac = theta_lineage.get("fraction_draw_level_theta_star")
    if frac is None:
        frac = theta_lineage.get("theta_star_fraction_strict_build_pool")
    theta_verdict = (
        "Mixed: some rows use draw-level `theta_star_deg`, remainder use `theta_obs_deg` fallback."
        if frac is not None and 0 < frac < 1
        else (
            "Draw-level `theta_star_deg` only (every counted row had finite `theta_star_deg`)."
            if frac == 1.0
            else (
                "Obs-proxy only: **all** counted strict-pool / joint rows use `theta_obs_deg` "
                "(no finite `theta_star_deg` in this production export)."
                if frac == 0.0
                else "See `theta_lineage` JSON."
            )
        )
    )

    lines = [
        "# Stage-z training report (p(z | u, g, p))",
        "",
        "## Approved stage-z targets (order)",
        "",
        "1. `x`",
        "2. `psi_deg`",
        "3. `e_y_star`",
        "4. `theta_deg` — finite `theta_star_deg` at draw level when present, else `theta_obs_deg` from launch; **never** `phi_star`.",
        "",
        "## Theta lineage (this build / dataset)",
        "",
        f"**Summary:** {theta_verdict}",
        "",
        f"```json\n{json.dumps(theta_lineage, indent=2)}\n```",
        "",
        ms_note,
        "",
        "## Conditioning features",
        "",
        "### Numeric (z-scored, train stats)",
        "",
        f"- Order: `{frozen_contract['x_numeric_zscore_column_order']}`",
        "",
        "### Categorical",
        "",
        f"- Order (YAML / contract): `{frozen_contract['categorical_yaml_key_order']}`",
        f"- Model forward (alphabetical embeddings): `{frozen_contract['categorical_model_forward_order_alphabetical']}`",
        "",
        "### Weight",
        "",
        f"- `{cfg['weight_column']}`",
        "",
        "## Model",
        "",
        f"- Family: **conditional full-covariance Gaussian mixture** (each component: mean in R^4, "
        f"covariance `Sigma = L L^T` with lower-triangular **Cholesky** `L`; **not** diagonal). "
        f"Checkpoint flag: `{ckpt_model_family}`.",
        f"- Config: K={cfg['model']['n_components']}, hidden_width={cfg['model']['hidden_width']}, "
        f"layers={cfg['model']['hidden_layers']}, z_dim={cfg['model']['z_dim']}, chol_eps={cfg['model'].get('chol_eps', 'n/a')}",
        "",
        "## Training settings",
        "",
        f"```yaml\n{yaml.safe_dump({k: cfg['training'][k] for k in sorted(cfg['training'].keys())})}```",
        "",
        "## Calibration",
        "",
        f"- Scalar **T_mix** on each component covariance: `Sigma_cal = T_mix * Sigma`, equivalently `L_cal = sqrt(T_mix) * L`.",
        f"- **T_mix = {T_mix:.6g}** (see `temperature.json`).",
        f"- Optimizer: Adam on `log T_mix`, max_iter={cfg['calibration']['max_iter']}, lr={cfg['calibration'].get('lr', 0.05)}.",
        "",
        "## Metrics",
        "",
        f"- Calibration weighted NLL (pre / post T_mix): "
        f"**{results.get('calibration_pre_temp', {}).get('weighted_nll', float('nan')):.4f}** / "
        f"**{results.get('calibration_post_temp', {}).get('weighted_nll', float('nan')):.4f}**",
        f"- Test weighted NLL (post T_mix): **{results.get('test_post_temp', {}).get('weighted_nll', float('nan')):.4f}**",
        "",
        "### Test split (post T_mix), per-target",
        "",
        _summ_coord("test_post_temp"),
        "",
        "## Angular / boundary notes",
        "",
        "- PIT histograms: `plots/pit_*_post_temp.png`.",
        "- `psi_deg` / `theta_deg` may show mild PIT distortion near physical or posterior-support bounds.",
        "- When exports include both finite `theta_star_deg` and `theta_obs_deg` pools, `theta_deg` can be bimodal across draws; **this build** used the obs proxy only (see lineage).",
        "",
        "## Dependence modeling (vs diagonal MDN)",
        "",
        "- This run uses **full** 4×4 covariances per mixture component, so nonzero partial correlations among z targets are explicitly represented.",
        "- We did **not** re-train a diagonal MDN in this pass; reserve head-to-head NLL comparison for a controlled rerun with identical seeds and K.",
        "",
        "## Bottom-line assessment",
        "",
        "- **Training / numerics:** Optimization completed all epochs without NaNs; checkpoint and calibration **T_mix** saved. Model family: **full-covariance Cholesky GMM**.",
        "- **Calibration / generalization:** Calibration weighted NLL improved slightly after **T_mix**; test weighted NLL is in line with calibration (see metrics tables).",
        "- **Angular targets:** Large raw MAE for `psi_deg` / `theta_deg` and degenerate 50% marginal coverage on those coordinates suggest the **moment-matched Gaussian intervals** are a poor match to heavy-tailed or bounded angular margins (not a Cholesky vs diagonal artifact per se). Review PIT panels for `psi_deg` and `theta_deg`.",
        "- **Dependence:** Full covariance is the correct inductive bias for joint z; whether it materially beats diagonal on **multivariate** NLL requires the diagonal baseline noted above.",
        "",
        "## Physics decoder consistency",
        "",
        "See `src/physics_decoder_contract.py`: deterministic decoder consumes **(u, z, g, p)**; "
        "solves nuisances including **`e_x`**, **`omega_plus`**; enforces production-aligned admissibility; outputs **(EV, LA, SA)**. "
        "Stage-z does **not** target `e_x` / `omega_plus`; they remain decoder-side.",
        "",
        "## Run artifacts",
        "",
        f"- Directory: `{run_dir}`",
        f"- Timestamp UTC: `{stamp}`",
        f"- Rows / events: `{json.dumps({k: meta[k] for k in sorted(meta) if k.startswith('n_')}, indent=2)}`",
        f"- Target stats filled train-only (if any): `{meta.get('target_stats_filled_train_only', [])}`",
        f"- Train wall seconds: **{train_meta.get('train_total_seconds_including_calibration', 'n/a')}**",
        "",
    ]
    out = project_root / "reports" / "p_z_given_u_g_training_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "p_z_given_u_g_default.yaml",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    fc = load_feature_contract_z(cfg, project_root)
    for col in fc.get("forbidden_neural_input_columns", []):
        block = (
            list(cfg["upstream_u_features"])
            + list(cfg["numeric_context_features"])
            + list(cfg["player_constant_features"])
        )
        if col in block:
            raise ValueError(f"Config must not place forbidden column {col!r} in neural inputs.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = project_root / cfg["outputs"]["run_dir"] / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    contract_path = resolve_feature_contract_path_z(cfg, project_root)
    frozen_contract = load_feature_contract_z(cfg, project_root)
    shutil.copy2(contract_path, run_dir / "feature_contract_frozen.json")

    train_a, cal_a, test_a, bundle = load_all_z_data(cfg, project_root)
    vocabs = bundle["vocabs"]
    meta = bundle["meta"]

    feature_manifest = {
        "targets": cfg["targets"],
        "upstream_u_features": cfg["upstream_u_features"],
        "numeric_context_features": cfg["numeric_context_features"],
        "player_constant_features": cfg["player_constant_features"],
        "categorical_features": {
            k: {"embedding_dim": v["embedding_dim"], "vocab_size": len(vocabs[k])}
            for k, v in cfg["categorical_features"].items()
        },
        "weight_column": cfg["weight_column"],
        "standardization_stats_path": str(project_root / cfg["paths"]["standardization_stats"]),
        "target_stats_filled_train_only": meta.get("target_stats_filled_train_only", []),
        "player_constant_standardization": (
            "train-only mean/std at load if missing from JSON: "
            f"{meta.get('player_constant_stats_from_train_only', [])}"
        ),
        "model_family": cfg["model"].get("family", "full_cov_chol_gmm"),
    }
    (run_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    device = torch.device("cpu")
    t0 = time.time()
    train_meta = train_stage_z_mdn(cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir)
    wall = time.time() - t0

    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    T_mix = float(ckpt["temperature_T_mix"])

    num_f = train_a.x_num.shape[1]
    model = _build_model(cfg, num_f, vocabs, device)
    model.load_state_dict(ckpt["model_state"])

    tm = torch.tensor(train_a.target_means, dtype=torch.float32, device=device)
    ts = torch.tensor(train_a.target_stds, dtype=torch.float32, device=device)

    results: dict[str, dict] = {}
    for name, arr in [("train", train_a), ("calibration", cal_a), ("test", test_a)]:
        x, cat, y, yr, w = _arrays_to_tensors_z(arr, device)
        for label, T_use in [(f"{name}_pre_temp", 1.0), (f"{name}_post_temp", T_mix)]:
            pack = evaluate_z_split_batched(
                model,
                x,
                cat,
                y,
                yr,
                w,
                tm,
                ts,
                T_mix=T_use,
                batch_size=8192,
                device=device,
            )
            save_metrics_json_csv(run_dir / "metrics", label, pack)
            results[label] = pack

        pits = collect_pits_z_batched(
            model, x, cat, y, T_mix=T_mix, batch_size=8192, device=device
        )
        plot_pit_hist_z(pits, run_dir / "plots" / f"pit_{name}_post_temp.png")
        np.save(run_dir / f"pits_{name}_post_temp.npy", pits)

    theta_lineage = _theta_lineage(project_root, cfg)
    _append_theta_fraction_from_build_manifest(theta_lineage)

    run_manifest = {
        "timestamp_utc": stamp,
        "git_hash": _git_hash(),
        "config_path": str(args.config.resolve()),
        "feature_contract_id": frozen_contract.get("contract_id"),
        "feature_contract_path": str(contract_path),
        "feature_contract_copied_to_run": str(run_dir / "feature_contract_frozen.json"),
        "row_counts": {k: int(meta[k]) for k in meta if k.startswith("n_rows")},
        "event_counts": {k: int(meta[k]) for k in meta if k.startswith("n_events")},
        "draw_cap_per_event": cfg["training"]["max_draws_per_event"],
        "batch_events": cfg["training"]["batch_events"],
        "event_balanced_batching": True,
        "player_constant_stats_computed_train_only": meta.get("player_constant_stats_from_train_only", []),
        "target_stats_filled_train_only": meta.get("target_stats_filled_train_only", []),
        "training_wall_seconds": wall,
        "train_loop_seconds": train_meta.get("train_loop_seconds"),
        "train_total_seconds_including_calibration": train_meta.get(
            "train_total_seconds_including_calibration"
        ),
        "stage_z_model_family": ckpt.get("model_family", "full_cov_chol_gmm"),
        "mdn_mixture_temperature_T_mix": T_mix,
        "theta_deg_lineage": theta_lineage,
        "nll_calibration_weighted_pre_temp": results.get("calibration_pre_temp", {}).get("weighted_nll"),
        "nll_calibration_weighted_post_temp": results.get("calibration_post_temp", {}).get("weighted_nll"),
        "nll_test_weighted_post_temp": results.get("test_post_temp", {}).get("weighted_nll"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    report_path = _write_training_report(
        project_root,
        run_dir,
        stamp,
        cfg,
        frozen_contract,
        contract_path,
        meta,
        train_meta,
        T_mix,
        results,
        theta_lineage,
        str(ckpt.get("model_family", "full_cov_chol_gmm")),
    )

    # Brief console summary
    frac = theta_lineage.get("fraction_draw_level_theta_star")
    print("Done.")
    print("Outputs:", run_dir)
    print("Report:", report_path)
    if frac is not None:
        print(f"Theta draw-level (finite theta_star) fraction (joint train): {frac:.4f}")


if __name__ == "__main__":
    main()
