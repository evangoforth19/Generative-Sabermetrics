Robustness Simulator
====================

Purpose
-------
Runs the frozen SBI forward simulator (stage-u × stage-z × physics decoder with
decoder admissibility="lenient") for every hitter × every pitch-context mixture
cluster (57 by default). Each cluster samples the seven pitch-context Gaussian
mixture dimensions; remaining decoder fields come from one template row in that
cluster (from sbi_context_event_master). Draws accumulate until EV, LA, SA, and
xwOBAcon pass a split-window stability rule, or `--max-admissible` samples,
or `--max-outer-iters`, whichever hits first.

Prerequisites
-------------
Use the project venv (arviz pulled in by MCMC tooling + torch + repo deps):

  /home/evangoforth03/Bayesian Research/MCMC 2/.venv_prod/bin/python run_robustness_simulator.py --help

Value model loads from `outputs/batted_ball_value_surface/` when `model_lgbm.pkl`
exists; otherwise a cached HistGradientBoosting surrogate trains once to
`artifacts/value_hgb_surrogate.pkl`.

Outputs
-------
artifacts/cluster_manifest.json   — cluster → pitch_group / component ids
artifacts/run_manifest.json     — knobs used
artifacts/value_model_meta.json   — which value backend ran
results/parquet/{hitter}_robustness_draws.parquet|csv — long-form draws
results/summary/{hitter}_cluster_summary.csv — per-cluster mean/std + convergence flags

Full run (expensive)
--------------------
Default settings target up to 8000 admissible decodes per (hitter × cluster).
57 × N_hitters × ~6k attempts can take many hours on CPU; use GPU if available:

  ../MCMC\ 2/.venv_prod/bin/python run_robustness_simulator.py --device cuda

Smoke / debug
------------
  ../MCMC\ 2/.venv_prod/bin/python run_robustness_simulator.py \
    --hitters-max 1 --clusters-max 2 --min-admissible 50 --max-admissible 120
