#!/usr/bin/env python3
"""Train stage-z hybrid: 2D full-cov Gaussian (x, e_y*) + von Mises angles; shared mixture weights."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.scripts.train_p_z_given_u_g import (  # noqa: E402
    _append_theta_fraction_from_build_manifest,
    _git_hash,
    _theta_lineage,
)
from sbi_forward_sim.src.data_z import ZArraysNativeCircular, load_all_z_data_native_circular  # noqa: E402
from sbi_forward_sim.src.eval_z_native_circular import (  # noqa: E402
    evaluate_z_native_circular_split_batched,
    plot_pit_hist_native,
    save_metrics_json_csv,
)
from sbi_forward_sim.src.feature_contract_z import (  # noqa: E402
    load_feature_contract_z,
    resolve_feature_contract_path_z,
)
from sbi_forward_sim.src.models_z import ConditionalHybridGaussVonMisesMixtureZ  # noqa: E402
from sbi_forward_sim.src.schema import Z_TARGET_COLUMNS  # noqa: E402
from sbi_forward_sim.src.train_z import train_stage_z_hybrid_native  # noqa: E402


def _save_metrics_strip(pack: dict, out_dir: Path, name: str) -> None:
    p = dict(pack)
    p.pop("_pit_arrays", None)
    save_metrics_json_csv(out_dir / "metrics", name, p)


def _slice_z_native_circular(a: ZArraysNativeCircular, idx: np.ndarray) -> ZArraysNativeCircular:
    return ZArraysNativeCircular(
        x_num=a.x_num[idx],
        cat={k: v[idx] for k, v in a.cat.items()},
        y_gauss=a.y_gauss[idx],
        psi_rad=a.psi_rad[idx],
        theta_rad=a.theta_rad[idx],
        y_raw_four=a.y_raw_four[idx],
        w=a.w[idx],
        event_id=a.event_id[idx],
        gauss_means=a.gauss_means,
        gauss_stds=a.gauss_stds,
        feature_names=a.feature_names,
    )


def _build_hybrid(cfg: dict, num_f: int, vocabs: dict[str, dict[str, int]], device: torch.device):
    mcfg = cfg["model"]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "p_z_given_u_g_native_circular.yaml",
    )
    ap.add_argument(
        "--resume-eval",
        type=Path,
        default=None,
        help="Existing run directory with checkpoint.pt: skip training, run eval/metrics/report only.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume_eval is not None:
        run_dir = args.resume_eval.resolve()
        cfg_path = run_dir / "config_resolved.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        stamp = run_dir.name
        fc_path = run_dir / "feature_contract_frozen.json"
        frozen_contract = json.loads(fc_path.read_text(encoding="utf-8"))
        train_meta = {}
    else:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if cfg.get("target_parameterization") != "native_circular_vm":
            raise ValueError("This script expects target_parameterization: native_circular_vm")

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

    train_a, cal_a, test_a, bundle = load_all_z_data_native_circular(cfg, project_root)
    vocabs = bundle["vocabs"]
    meta = bundle["meta"]
    nt_manifest = meta.get("native_target_manifest", {})
    if args.resume_eval is None:
        (run_dir / "target_transform_manifest.json").write_text(json.dumps(nt_manifest, indent=2), encoding="utf-8")

        feature_manifest = {
            "raw_targets_contract": cfg["targets"],
            "target_parameterization": "native_circular_vm",
            "internal_training_targets_note": "y_gauss std (x,e_y*); psi_rad, theta_rad wrapped [-pi,pi)",
            "upstream_u_features": cfg["upstream_u_features"],
            "numeric_context_features": cfg["numeric_context_features"],
            "player_constant_features": cfg["player_constant_features"],
            "categorical_features": {
                k: {"embedding_dim": v["embedding_dim"], "vocab_size": len(vocabs[k])}
                for k, v in cfg["categorical_features"].items()
            },
            "weight_column": cfg["weight_column"],
            "standardization_stats_path": str(project_root / cfg["paths"]["standardization_stats"]),
            "model_family": cfg["model"].get("family", "hybrid_gauss_vonmises_shared"),
        }
        (run_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

        train_meta = train_stage_z_hybrid_native(cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir)

    try:
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    T_gauss = float(ckpt["temperature_T_gauss"])
    T_ang = float(ckpt["temperature_T_ang"])
    T_gx = ckpt.get("temperature_T_gauss_x")
    T_gy = ckpt.get("temperature_T_gauss_y")
    if T_gx is None or T_gy is None:
        T_gx_f: float | None = None
        T_gy_f: float | None = None
    else:
        T_gx_f = float(T_gx)
        T_gy_f = float(T_gy)

    num_f = train_a.x_num.shape[1]
    model = _build_hybrid(cfg, num_f, vocabs, device)
    model.load_state_dict(ckpt["model_state"])

    gm = torch.tensor(train_a.gauss_means, dtype=torch.float32, device=device)
    gs = torch.tensor(train_a.gauss_stds, dtype=torch.float32, device=device)

    n_samp = int(cfg.get("eval", {}).get("n_predictive_samples", 256))
    bs = int(cfg.get("eval", {}).get("circular_eval_batch_size", 512))
    eval_seed = int(cfg["training"].get("seed", 42))
    max_tr_eval = int(cfg.get("eval", {}).get("max_rows_train_eval", 0) or 0)
    train_eval_a = train_a
    if max_tr_eval > 0 and train_a.x_num.shape[0] > max_tr_eval:
        rng = np.random.default_rng(eval_seed + 99)
        idx = rng.choice(train_a.x_num.shape[0], size=max_tr_eval, replace=False)
        train_eval_a = _slice_z_native_circular(train_a, idx)

    results: dict[str, dict] = {}
    for name, arr in [("train", train_eval_a), ("calibration", cal_a), ("test", test_a)]:
        x = torch.from_numpy(arr.x_num).to(device)
        cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
        yg = torch.from_numpy(arr.y_gauss).to(device)
        psi = torch.from_numpy(arr.psi_rad).to(device)
        theta = torch.from_numpy(arr.theta_rad).to(device)
        yr = torch.from_numpy(arr.y_raw_four).to(device)
        w = torch.from_numpy(arr.w).to(device)
        for label, tg, ta, tgx, tgy in [
            (f"{name}_pre_temp", 1.0, 1.0, None, None),
            (f"{name}_post_temp", T_gauss, T_ang, T_gx_f, T_gy_f),
        ]:
            pack = evaluate_z_native_circular_split_batched(
                model,
                x,
                cat,
                yg,
                psi,
                theta,
                yr,
                w,
                gm,
                gs,
                T_gauss=tg,
                T_ang=ta,
                T_gauss_x=tgx,
                T_gauss_y=tgy,
                n_samples=n_samp,
                batch_size=bs,
                device=device,
                seed=eval_seed + {"train": 0, "calibration": 1, "test": 2}[name],
            )
            pits = pack.pop("_pit_arrays", None)
            _save_metrics_strip(pack, run_dir, label)
            results[label] = pack
            if pits is not None and label.endswith("_post_temp"):
                plot_pit_hist_native(
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

    def _load_metrics(rel: str) -> dict | None:
        p = (project_root / rel).resolve()
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    canon = cfg.get("experiment", {}).get("reference_circular_canonical", "")
    pacc = cfg.get("experiment", {}).get("reference_pointacc_run", "")
    m_can = _load_metrics(f"{canon}/metrics/test_post_temp_metrics.json") if canon else None
    m_pa = _load_metrics(f"{pacc}/metrics/test_post_temp_metrics.json") if pacc else None
    te = results.get("test_post_temp", {})

    report_lines = [
        "# Stage-z native circular likelihood experiment (p(z|u,g,p))",
        "",
        "## 1. Raw target semantics (unchanged)",
        "",
        f"- Contract order: `{list(Z_TARGET_COLUMNS)}`",
        "- **theta_deg**: finite `theta_star_deg` when present at draw level, else `theta_obs_deg`; never `phi_star`.",
        "- **e_x**, **omega_plus** are not stage-z neural targets.",
        "",
        "## 2. Internal training representation",
        "",
        "- **Gaussian block:** `(x, e_y_star)` standardized with train JSON stats (same as linear stage-z).",
        "- **Angles:** `psi_deg`, `theta_deg` from disk → radians, wrapped to **[-π, π)**.",
        "- **Likelihood:** ∑_k π_k(u,g,p) · N(μ_k^g, L_k L_k^T) · VM(ψ|μ_k^ψ, κ_k^ψ) · VM(θ|μ_k^θ, κ_k^θ) with **shared** π_k.",
        "",
        "```json",
        json.dumps(nt_manifest, indent=2),
        "```",
        "",
        "## 3. Theta provenance (this build)",
        "",
        f"```json\n{json.dumps(theta_lineage, indent=2)}\n```",
        "",
        "## 4. Model & dependence",
        "",
        "- **Family:** hybrid Gaussian (2D full covariance, Cholesky) + independent von Mises per angle per component.",
        "- **Dependence:** single categorical posterior π_k(u) couples all four target factors within component k.",
        "",
        "## 5. Training settings",
        "",
        f"```yaml\n{yaml.safe_dump({'training': cfg['training'], 'model': cfg['model']})}\n```",
        "",
        "## 6. Calibration",
        "",
    ]
    if T_gx_f is not None and T_gy_f is not None:
        report_lines.extend(
            [
                f"- **Anisotropic Gaussian block:** `T_gauss_x={T_gx_f:.6g}`, `T_gauss_y={T_gy_f:.6g}` — "
                "row `i` of `L` scaled by `sqrt(T_i)` (equivalently Σ_cal = D Σ D, D=diag(√T_x,√T_y)).",
                f"- **T_gauss** (geometric mean √(T_x T_y)): **{T_gauss:.6g}** — legacy summary scalar.",
                f"- **T_ang:** {T_ang:.6g} — `κ_cal = κ / T_ang` for both von Mises factors.",
            ]
        )
    else:
        report_lines.extend(
            [
                f"- **T_gauss:** {T_gauss:.6g} — `Sigma_cal = T_gauss Sigma`, `L_cal = sqrt(T_gauss) L`.",
                f"- **T_ang:** {T_ang:.6g} — `κ_cal = κ / T_ang` for both von Mises factors.",
            ]
        )
    report_lines.extend(
        [
            "",
            "## 7. Test metrics (post-calibration)",
            "",
            f"- Hybrid weighted NLL (joint, this parameterization): **{te.get('weighted_nll_hybrid_natural_space', float('nan')):.4f}**",
            f"- `x` MAE / RMSE: **{te.get('mae_raw_x', float('nan')):.4f}** / **{te.get('rmse_raw_x', float('nan')):.4f}**",
            f"- `e_y_star` MAE / RMSE: **{te.get('mae_raw_e_y_star', float('nan')):.4f}** / **{te.get('rmse_raw_e_y_star', float('nan')):.4f}**",
            f"- `psi_deg` wrapped MAE / RMSE: **{te.get('wrapped_mae_deg_psi', float('nan')):.4f}°** / **{te.get('wrapped_rmse_deg_psi', float('nan')):.4f}°**",
            f"- `theta_deg` wrapped MAE / RMSE: **{te.get('wrapped_mae_deg_theta', float('nan')):.4f}°** / **{te.get('wrapped_rmse_deg_theta', float('nan')):.4f}°**",
            "",
            "### Sample-based coverage (marginal)",
            "",
            f"```json\n{json.dumps({k: te.get(k) for k in ['coverage_50_marginal_sample', 'coverage_80_marginal_sample', 'coverage_90_marginal_sample'] if k in te}, indent=2)}\n```",
            "",
            "- PIT histograms: `plots/pit_test_post_temp.png`",
            "",
            "## 8. Comparison vs reference runs (test, post-temp)",
            "",
            "| Metric | Canonical circular 6D GMM `183758Z` | Point-acc `184405Z` | **This run (native VM)** |",
            "|---|---:|---:|---:|",
        ]
    )

    def _scalar(m: dict | None, key: str) -> float:
        if not m or key not in m:
            return float("nan")
        return float(m[key])

    def _cov(m: dict | None, coord: str) -> float:
        if not m:
            return float("nan")
        d = m.get("coverage_50_marginal_sample")
        if not isinstance(d, dict):
            return float("nan")
        return float(d.get(coord, float("nan")))

    report_lines.extend(
        [
            f"| cov50 psi | {_cov(m_can, 'psi_deg'):.4f} | {_cov(m_pa, 'psi_deg'):.4f} | {_cov(te, 'psi_deg'):.4f} |",
            f"| cov50 theta | {_cov(m_can, 'theta_deg'):.4f} | {_cov(m_pa, 'theta_deg'):.4f} | {_cov(te, 'theta_deg'):.4f} |",
            f"| MAE ψ (wrapped) | {_scalar(m_can, 'wrapped_mae_deg_psi'):.3f} | {_scalar(m_pa, 'wrapped_mae_deg_psi'):.3f} | {_scalar(te, 'wrapped_mae_deg_psi'):.3f} |",
            f"| MAE θ (wrapped) | {_scalar(m_can, 'wrapped_mae_deg_theta'):.3f} | {_scalar(m_pa, 'wrapped_mae_deg_theta'):.3f} | {_scalar(te, 'wrapped_mae_deg_theta'):.3f} |",
            f"| MAE x | {_scalar(m_can, 'mae_raw_x'):.4f} | {_scalar(m_pa, 'mae_raw_x'):.4f} | {_scalar(te, 'mae_raw_x'):.4f} |",
            f"| MAE e_y* | {_scalar(m_can, 'mae_raw_e_y_star'):.4f} | {_scalar(m_pa, 'mae_raw_e_y_star'):.4f} | {_scalar(te, 'mae_raw_e_y_star'):.4f} |",
            "",
            "**Note:** NLL across rows is not directly comparable (6D circular-GMM vs hybrid joint).",
            "",
        ]
    )

    cov_psi = _cov(te, "psi_deg")
    cov_th = _cov(te, "theta_deg")
    c_can_psi = _cov(m_can, "psi_deg")
    c_can_th = _cov(m_can, "theta_deg")
    better_cal = (cov_psi + cov_th) >= (c_can_psi + c_can_th) - 1e-6 and _scalar(te, "mae_raw_x") <= _scalar(
        m_can, "mae_raw_x"
    ) + 0.5

    verdict = (
        "**Candidate for new canonical:** native VM improves angular coverage vs `183758Z` without materially hurting x/e_y*."
        if better_cal and cov_psi >= c_can_psi and cov_th >= c_can_th
        else "**Not promoted:** does not clearly beat canonical circular GMM `183758Z` on angular calibration + non-angular MAE."
    )
    report_lines.extend(["## 9. Bottom line", "", verdict, "", f"- Run directory: `{run_dir}`", ""])

    report_path = project_root / "reports" / "p_z_given_u_g_native_circular_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    resolved_cfg_path = str((run_dir / "config_resolved.yaml").resolve())
    run_manifest = {
        "timestamp_utc": stamp,
        "git_hash": _git_hash(),
        "config_path": resolved_cfg_path,
        "feature_contract_id": frozen_contract.get("contract_id"),
        "target_parameterization": "native_circular_vm",
        "model_family": "hybrid_gauss_vonmises_shared",
        "row_counts": {k: int(meta[k]) for k in meta if k.startswith("n_rows")},
        "event_counts": {k: int(meta[k]) for k in meta if k.startswith("n_events")},
        "temperature_T_gauss": T_gauss,
        "temperature_T_gauss_x": T_gx_f,
        "temperature_T_gauss_y": T_gy_f,
        "temperature_T_ang": T_ang,
        "theta_deg_lineage": theta_lineage,
        "nll_test_hybrid_weighted_post_temp": te.get("weighted_nll_hybrid_natural_space"),
        "reference_circular_canonical": str(canon),
        "reference_pointacc_run": str(pacc),
        "eval_predictive_samples": n_samp,
        "eval_train_rows_used": int(train_eval_a.x_num.shape[0]),
        "eval_train_rows_total": int(train_a.x_num.shape[0]),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print("Done. Outputs:", run_dir)
    print("Report:", report_path)
    print(
        f"Test wrapped MAE psi/theta: {te.get('wrapped_mae_deg_psi', float('nan')):.3f}, "
        f"{te.get('wrapped_mae_deg_theta', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
