# Data inclusion notes

## Included
- Processed SBI train/cal/test splits (`02_sbi_forward_sim/data_processed/`)
- Frozen stage-u / stage-z checkpoints used by robustness + bootstrap pipelines
- LightGBM xwOBAcon value surface
- Pitch-context GMM fits and May 8 robustness / paper figure outputs
- Physics EV calibrator (`ev_calibrator.joblib`) from 20260508_0418Z
- Twelve-hitter empirical BIP parquet for overlays
- Metric Folder full G dataset used for value-model training

## Intentionally excluded (redundant / regenerable / over GitHub limits)
- `sbi_draws_basic.parquet` and `sbi_draws_raw.parquet` (~138MB each; regenerable via `build_sbi_datasets.py`)
- Robustness CSV draw duplicates (parquet kept)
- Older non-calibrated `Robustness Simulator/results/`
- `.venv_prod`, exploratory MCMC notebooks, heldout predictive-draw dumps
- Statcast chunk cache / pybaseball_cache (~1.5GB)
- Physics calibrator training draw dumps (~190MB); model artifact retained
