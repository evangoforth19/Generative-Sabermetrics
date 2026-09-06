#!/usr/bin/env python3
"""Train stage-z with circular (sin/cos) angular targets; full-covariance GMM in R^6."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.scripts.train_p_z_given_u_g import (  # noqa: E402
    _append_theta_fraction_from_build_manifest,
    _build_model,
    _git_hash,
    _theta_lineage,
)
from sbi_forward_sim.src.data_z import load_all_z_data_circular  # noqa: E402
from sbi_forward_sim.src.eval_z_circular import (  # noqa: E402
    evaluate_z_circular_split_batched,
    plot_pit_hist_circular,
    save_metrics_json_csv,
)
from sbi_forward_sim.src.feature_contract_z import (  # noqa: E402
    load_feature_contract_z,
    resolve_feature_contract_path_z,
)
from sbi_forward_sim.src.target_transform_z import CIRCULAR_SIX_ORDER, RAW_FOUR_ORDER  # noqa: E402
from sbi_forward_sim.src.train_z import train_stage_z_mdn  # noqa: E402


def _save_metrics_strip_internal(pack: dict, out_dir: Path, name: str) -> None:
    p = dict(pack)
    p.pop("_pit_arrays", None)
    save_metrics_json_csv(out_dir / "metrics", name, p)


def _write_circular_report(
    project_root: Path,
    run_dir: Path,
    stamp: str,
    cfg: dict,
    frozen_contract: dict,
    meta: dict,
    train_meta: dict,
    T_mix: float,
    results: dict,
    theta_lineage: dict,
    baseline_test_metrics: dict | None,
    baseline_run_label: str,
) -> Path:
    circ = meta.get("circ_target_manifest", {})
    te = results.get("test_post_temp", {})
    bmae = baseline_test_metrics.get("mae_raw") if baseline_test_metrics else None
    bnll = baseline_test_metrics.get("weighted_nll") if baseline_test_metrics else None
    bx = bmae[0] if isinstance(bmae, list) and len(bmae) > 0 else None
    bp = bmae[1] if isinstance(bmae, list) and len(bmae) > 1 else None
    bey = bmae[2] if isinstance(bmae, list) and len(bmae) > 2 else None
    bth = bmae[3] if isinstance(bmae, list) and len(bmae) > 3 else None

    lines = [
        "# Stage-z circular target experiment (p(z|u,g,p))",
        "",
        "## 1. Raw stage-z targets (contract / parquet semantics)",
        "",
        f"- Order: `{list(RAW_FOUR_ORDER)}`",
        "- **theta_deg** definition unchanged: finite `theta_star_deg` when present on draws, else `theta_obs_deg`; never `phi_star`.",
        "",
        "## 2. Internal training targets (6D)",
        "",
        f"- Order: `{list(CIRCULAR_SIX_ORDER)}`",
        "- Angles in degrees → radians, then `sin`, `cos` for `psi` and `theta`.",
        "- `x` and `e_y_star` use the same mean/std as `standardization_stats.json` for those columns; "
        "`sin_*` / `cos_*` use **train-split** mean and std after the trig transform.",
        "",
        "```json",
        json.dumps(circ, indent=2),
        "```",
        "",
        "## 3. Theta provenance (this build)",
        "",
        f"```json\n{json.dumps(theta_lineage, indent=2)}\n```",
        "",
        "## 4. Model family",
        "",
        "- **Conditional full-covariance Gaussian mixture** on the **6D standardized** circular block.",
        "- **Dependence:** single Cholesky factor per mixture component in **R^6** (not diagonal, not independent heads).",
        "",
        "## 5–7. Training & calibration",
        "",
        f"- Training: `{yaml.safe_dump({k: cfg['training'][k] for k in sorted(cfg['training'].keys())})}`",
        f"- Calibration: scalar **T_mix** with `L_cal = sqrt(T_mix) * L` (same as linear run). **T_mix = {T_mix:.6g}**.",
        "",
        "## 8–10. Test metrics (raw space for `x`, `e_y_star`; wrapped angles)",
        "",
        "### 8–9. Scalar summaries (test, post T_mix)",
        "",
        f"- Weighted **6D standardized** NLL (not comparable to 4D linear run): **{te.get('weighted_nll_6d_std_space', float('nan')):.4f}**",
        f"- `x` MAE / RMSE (raw): **{te.get('mae_raw_x', float('nan')):.4f}** / **{te.get('rmse_raw_x', float('nan')):.4f}**",
        f"- `e_y_star` MAE / RMSE (raw): **{te.get('mae_raw_e_y_star', float('nan')):.4f}** / **{te.get('rmse_raw_e_y_star', float('nan')):.4f}**",
        f"- `psi_deg` **wrapped** MAE / RMSE: **{te.get('wrapped_mae_deg_psi', float('nan')):.4f}** / **{te.get('wrapped_rmse_deg_psi', float('nan')):.4f}**",
        f"- `theta_deg` **wrapped** MAE / RMSE: **{te.get('wrapped_mae_deg_theta', float('nan')):.4f}** / **{te.get('wrapped_rmse_deg_theta', float('nan')):.4f}**",
        "",
        "### 10. Sample-based marginal coverage (test)",
        "",
        f"```json\n{json.dumps({k: te.get(k) for k in ['coverage_50_marginal_sample', 'coverage_80_marginal_sample', 'coverage_90_marginal_sample'] if k in te}, indent=2)}\n```",
        "",
        "- PIT histograms: `plots/pit_*_post_temp.png`. Angular PITs use mixture samples → `atan2` → empirical CDF vs truth.",
        "",
        "## 11. Comparison vs raw-angle full-cov run",
        "",
        f"- Baseline run reference: `{baseline_run_label}`",
        "",
        "| Metric (test, post T) | Linear 4D baseline (naive ° error) | Circular 6D (this run) |",
        "|---|---:|---:|",
        f"| NLL (native) | {bnll} (4D std) | {te.get('weighted_nll_6d_std_space', float('nan')):.4f} (6D std) |",
        f"| MAE x | {bx} | {te.get('mae_raw_x', float('nan')):.4f} |",
        f"| MAE e_y* | {bey} | {te.get('mae_raw_e_y_star', float('nan')):.4f} |",
        f"| MAE psi (baseline: **linear** °) | {bp} | **{te.get('wrapped_mae_deg_psi', float('nan')):.4f}** (wrapped) |",
        f"| MAE theta (baseline linear °) | {bth} | **{te.get('wrapped_mae_deg_theta', float('nan')):.4f}** (wrapped) |",
        "",
        "- **Interpretation:** baseline angular MAE is not wrapped; circular run uses **wrapped** angular error — fairer for angles. "
        "Compare PIT/coverage shape qualitatively between runs using the plots.",
        "",
        "## 12. Bottom line",
        "",
        "- See whether wrapped angular MAE/RMSE improved vs baseline linear MAE, whether angular coverage at 50/80/90% is closer to nominal, and whether angular PITs are less U-shaped.",
        "- `x` / `e_y_star` should be largely unaffected by the angular reparameterization (same marginal structure after inverse transform).",
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{run_dir}`",
        f"- `target_transform_manifest.json`, `run_manifest.json`, `metrics/`, `temperature.json`",
        "",
    ]
    out = project_root / "reports" / "p_z_given_u_g_circular_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "p_z_given_u_g_circular.yaml",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if cfg.get("target_parameterization") != "circular_6d":
        raise ValueError("This script expects target_parameterization: circular_6d in config.")

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

    train_a, cal_a, test_a, bundle = load_all_z_data_circular(cfg, project_root)
    vocabs = bundle["vocabs"]
    meta = bundle["meta"]
    circ_manifest = meta.get("circ_target_manifest", {})
    (run_dir / "target_transform_manifest.json").write_text(json.dumps(circ_manifest, indent=2), encoding="utf-8")

    feature_manifest = {
        "raw_targets_contract": cfg["targets"],
        "internal_training_targets": list(CIRCULAR_SIX_ORDER),
        "target_parameterization": "circular_6d",
        "upstream_u_features": cfg["upstream_u_features"],
        "numeric_context_features": cfg["numeric_context_features"],
        "player_constant_features": cfg["player_constant_features"],
        "categorical_features": {
            k: {"embedding_dim": v["embedding_dim"], "vocab_size": len(vocabs[k])}
            for k, v in cfg["categorical_features"].items()
        },
        "weight_column": cfg["weight_column"],
        "standardization_stats_path": str(project_root / cfg["paths"]["standardization_stats"]),
        "model_family": cfg["model"].get("family", "full_cov_chol_gmm_circular6"),
    }
    (run_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    device = torch.device("cpu")
    train_meta = train_stage_z_mdn(cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir)

    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    T_mix = float(ckpt["temperature_T_mix"])

    num_f = train_a.x_num.shape[1]
    model = _build_model(cfg, num_f, vocabs, device)
    model.load_state_dict(ckpt["model_state"])

    cm = torch.tensor(train_a.circ_means, dtype=torch.float32, device=device)
    cs = torch.tensor(train_a.circ_stds, dtype=torch.float32, device=device)

    n_samp = int(cfg.get("eval", {}).get("n_predictive_samples", 256))
    bs_circ = int(cfg.get("eval", {}).get("circular_eval_batch_size", 512))
    eval_seed = int(cfg["training"].get("seed", 42))

    results: dict[str, dict] = {}
    for name, arr in [("train", train_a), ("calibration", cal_a), ("test", test_a)]:
        x = torch.from_numpy(arr.x_num).to(device)
        cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
        y = torch.from_numpy(arr.y).to(device)
        yr4 = torch.from_numpy(arr.y_raw_four).to(device)
        w = torch.from_numpy(arr.w).to(device)
        for label, T_use in [(f"{name}_pre_temp", 1.0), (f"{name}_post_temp", T_mix)]:
            pack = evaluate_z_circular_split_batched(
                model,
                x,
                cat,
                y,
                yr4,
                w,
                cm,
                cs,
                T_mix=T_use,
                n_samples=n_samp,
                batch_size=bs_circ,
                device=device,
                seed=eval_seed + {"train": 0, "calibration": 1, "test": 2}[name],
            )
            pits = pack.pop("_pit_arrays", None)
            _save_metrics_strip_internal(pack, run_dir, label)
            results[label] = pack
            if pits is not None and T_use == T_mix:
                plot_pit_hist_circular(
                    {
                        "x": pits["x"],
                        "ey": pits["e_y_star"],
                        "psi": pits["psi_deg"],
                        "theta": pits["theta_deg"],
                    },
                    run_dir / "plots" / f"pit_{name}_post_temp.png",
                )
                np.savez_compressed(
                    run_dir / f"pits_{name}_post_temp.npz",
                    x=pits["x"],
                    e_y_star=pits["e_y_star"],
                    psi_deg=pits["psi_deg"],
                    theta_deg=pits["theta_deg"],
                )

    theta_lineage = _theta_lineage(project_root, cfg)
    _append_theta_fraction_from_build_manifest(theta_lineage)

    baseline_dir = project_root / cfg.get("experiment", {}).get("linear_baseline_run_dir", "")
    baseline_test = None
    if baseline_dir:
        bp = (project_root / baseline_dir / "metrics" / "test_post_temp_metrics.json").resolve()
        if bp.is_file():
            baseline_test = json.loads(bp.read_text(encoding="utf-8"))

    run_manifest = {
        "timestamp_utc": stamp,
        "git_hash": _git_hash(),
        "config_path": str(args.config.resolve()),
        "sweep_experiment_id": cfg.get("experiment", {}).get("sweep_id"),
        "reference_circular_canonical": cfg.get("experiment", {}).get("reference_circular_canonical"),
        "feature_contract_id": frozen_contract.get("contract_id"),
        "target_parameterization": "circular_6d",
        "internal_target_order": list(CIRCULAR_SIX_ORDER),
        "circ_target_manifest_path": str(run_dir / "target_transform_manifest.json"),
        "row_counts": {k: int(meta[k]) for k in meta if k.startswith("n_rows")},
        "event_counts": {k: int(meta[k]) for k in meta if k.startswith("n_events")},
        "mdn_mixture_temperature_T_mix": T_mix,
        "theta_deg_lineage": theta_lineage,
        "nll_calibration_6d_weighted_pre_temp": results.get("calibration_pre_temp", {}).get(
            "weighted_nll_6d_std_space"
        ),
        "nll_calibration_6d_weighted_post_temp": results.get("calibration_post_temp", {}).get(
            "weighted_nll_6d_std_space"
        ),
        "nll_test_6d_weighted_post_temp": results.get("test_post_temp", {}).get("weighted_nll_6d_std_space"),
        "baseline_linear_run": str(baseline_dir),
        "eval_predictive_samples": n_samp,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    report_path = _write_circular_report(
        project_root,
        run_dir,
        stamp,
        cfg,
        frozen_contract,
        meta,
        train_meta,
        T_mix,
        results,
        theta_lineage,
        baseline_test,
        str(baseline_dir),
    )

    te = results.get("test_post_temp", {})
    print("Done. Outputs:", run_dir)
    print("Report:", report_path)
    print(
        f"Test wrapped MAE psi/theta: {te.get('wrapped_mae_deg_psi', float('nan')):.3f}, "
        f"{te.get('wrapped_mae_deg_theta', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
