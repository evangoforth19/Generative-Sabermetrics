Final Inference Folder
======================

This folder contains the core scripts for the final inverse-collision inference stage and
the downstream SBI dataset build step.

Files
-----
1) run_mcmc_posterior_bank.py
   - Source physics/manifold script.
   - Provides hitter construction, event preprocessing helpers, sensor perturbation logic,
     and exact-root manifold functions used by production.

2) run_inverse_collision_production.py
   - Main production inference runner (event-by-event).
   - Performs exact-root admissibility-gated MCMC, writes per-event and aggregate outputs,
     and can export train/calibration artifacts.

3) build_sbi_datasets.py
   - Downstream dataset builder for SBI/forward-model training.
   - Consumes production outputs and writes processed split datasets.

Typical Run Order
-----------------
Step 1: Run production inference with run_inverse_collision_production.py
        (this creates production outputs under an output root).

Step 2: Run build_sbi_datasets.py against that production output root
        to generate processed train/calibration/test datasets for SBI.

Notes
-----
- run_inverse_collision_production.py references run_mcmc_posterior_bank.py as its source script.
- If changing output locations, keep paths consistent between production output and dataset builder input.
