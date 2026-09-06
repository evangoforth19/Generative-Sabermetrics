# Generative Sabermetrics

Physics-based simulation framework for baseball intelligence — companion repository for the working paper *Generative Sabermetrics: A Physics-Based Simulation Framework for Baseball Intelligence* (Evan Goforth, May 2026).

This is a **skinny production spine**: the code, processed datasets, frozen model checkpoints, and result artifacts that supply the paper — not the full exploratory VM workspace.

## Repository layout

| Path | Role |
|------|------|
| `01_inverse_inference/` | Exact-root inverse-collision MCMC (production runners + incumbent physics source) |
| `02_sbi_forward_sim/` | MDN stage-u / stage-z package, physics decoder, processed SBI datasets, frozen checkpoints |
| `03_value_model/` | LightGBM xwOBAcon value surface + Metric Folder training extract |
| `04_robustness_metrics/` | Robustness simulator, pitch-context GMMs, calibrated runs, paper figure outputs |
| `scripts/` | Shared pipeline scripts (GMM, robustness scores, bootstrap overlays, Statcast download, …) |
| `artifacts/physics_decoder_calibration/` | Frozen EV calibrator used in the May 8 calibrated runs |
| `data/statcast_pybaseball/` | Empirical BIP extract for overlays / bootstrap context |
| `paper/` | Working paper markdown + figures |

See `DATA_NOTES.md` for what was intentionally excluded (venvs, heldout dumps, regenerable giant intermediates).

## Paper pipeline (high level)

1. **Inverse latent inference** — sample admissible collision states \((x,\psi,e_y^\star,e_x,\omega^+)\).
2. **Build SBI datasets** — train/cal/test splits for upstream \(u\) and latent \(z\).
3. **Train generative kernels** — MDNs \(p_\theta(u\mid c)\) and \(p_\phi(z\mid u,c)\); deterministic physics decoder → (EV, LA, SA).
4. **Value model** — LightGBM maps (EV, LA, SA) → xwOBAcon.
5. **Simulation-based metrics** — pitch-context GMMs → forward simulate → robustness / vulnerability metrics and counterfactuals.

Frozen checkpoints used by default in the robustness / bootstrap scripts:

- stage-u: `02_sbi_forward_sim/outputs/p_u_given_g/20260407_200302Z` (robustness) and `…/20260407_193337Z` (bootstrap)
- stage-z: `…/p_z_given_u_g/20260501_055904Z` (robustness) and `…/20260501_055549Z` (bootstrap)

## Notes

- Private research repository.
- Paths inside some scripts still reflect the original VM layout (`Bayesian Research/…`); adjust or symlink when re-running locally.
- Raw MCMC draw banks `sbi_draws_{raw,basic}.parquet` were omitted (GitHub size); regenerate with `build_sbi_datasets.py` if needed.
