# Open decisions (SBI preprocessing)

This file reflects the **current** policies implemented under `sbi_forward_sim/`.

## Stage-z target vector (SBI forward simulator)

The **reduced** stage-2 target slice is **only**:

`x`, `psi_deg`, `e_y_star`, `theta_deg`

- **`theta_deg`:** per draw, **finite `theta_star_deg`** if present in production exports, **else `theta_obs_deg`** from `launch_angle_obs_deg`. Never `phi_star`.
- **`e_x`**, **`omega_plus`:** **not** stage-z NN targets; they remain on raw/joint/debug draws where exported and are **computed or solved inside the deterministic physics decoder** alongside production exact-root admissibility.

**Broader posterior-bank** train exports may still list a larger “latent” block (e.g. `x`, `psi`, `e_x`, `e_y_star`, `omega_plus`) for non-SBI workflows. That is **not** the SBI stage-z definition; see `schema.Z_TARGET_COLUMNS` and `modeling_schema.json`.

## θ naming on processed z tables

- Processed **`z_*.parquet`** carry **`theta_deg`** only (canonical), not `theta_obs_deg` / `theta_star_deg` as separate target columns in the stage-z slice (sources may remain on full `sbi_draws_*.parquet`).

## `release_speed` in g

- **Preferred:** **Statcast mph** merged from the project Statcast pickle (`batter_data_2020_2025.pkl` or `batter_data_2024_2025.pkl` under `MCMC 2/`, or `--statcast-pickle` on `scripts/build_sbi_datasets.py`).
- **Space-constrained default:** if no pickle is present and `context_event_table` has no `release_speed` column, the build sets **`release_speed` from ‖(vx0, vy0, vz0)‖** (ft/s → mph). Provenance is recorded as `derived_sqrt_vx0_vy0_vz0_ft_s_to_mph_no_statcast_pickle` in the run manifest / `input_discovery` extras. This is **not** identical to a keyed Statcast row merge; use the pickle when you need that parity.

## Semantic fallbacks

- **No** semantic fallbacks for **angles**: missing required inputs raise explicit errors.
- **`release_speed`:** pickle merge if available; else kinematics derivation as above (documented, not silent).
- **Parquet vs CSV** for a few train-export artifacts is the only allowed **format** fallback (see `discover_production_inputs`).

## Stage z screening

- **Primary** stage-z rows are built from **basic-screened** long draws (`posterior_draws_long_basic_screened.parquet`), event-split aligned with stage-u (`z_screen_choice: basic_screened_primary`).
- **Strict-screened** slices are written as **`z_strict_{train,calibration,test}.parquet`** for optional diagnostics, not as the sole training source.

## Physics decoder context `g` (vs neural `g`)

- **Neural stage-u / stage-z** feature packs use **`schema.G_COLUMNS`** (frozen training contracts; unchanged).
- The **deterministic physics decoder** (`decode_bip_from_sample`) uses a **strict superset** of pitch/kinematics context so incoming velocity and \(\omega^-\) match `run_mcmc_posterior_bank` (see `src/physics_decoder_contract.py` and **`manifests/physics_decoder_input_contract.json`**).
- **`release_pos_y` is formally required** at the decoder boundary (Statcast feet → `build_incoming_pitch_from_sensor`). Missing keys raise **`DecoderInputError`** before any physics or bank import side effects (after validation runs).
- **`schema.P_COLUMNS`** still includes `I0_oz_in2` for tabular joins; the decoder’s **`DECODER_P_REQUIRED`** subset is what `Hitter` reads (`I0` is derived internally from bat weight and \(r_g\)).

## Other notes

- **`phi_star` is never θ.** Spray-related fields stay separate; `d_tilde` is the stage‑u spray-direction scalar. Processed **u** datasets must **not** include `phi_star`.
- **Player constants** are medians per batter from `selected_events` rows that survive the bip_model + fair-wedge event filter.
- **Splits** are a fresh 70 / 15 / 15 event-level split (stratified approximately by batter).

## Provisional canonical training runs (lineage freeze)

These are the **default checkpoints** to use for SBI stack continuation unless a report explicitly supersedes them:

| Stage | Role | Run directory (under `sbi_forward_sim/`) |
|-------|------|-------------------------------------------|
| **Stage-u** \(p(u\mid g,p)\) | Provisional canonical | `outputs/p_u_given_g/20260407_173057Z/` |
| **Stage-z** \(p(z\mid u,g,p)\) | Provisional canonical (**native** Gaussian + von Mises, shared mixture weights) | `outputs/p_z_given_u_g/20260407_185923Z/` |

**Prior stage-z baseline (6D circular Gaussian on sin/cos targets), not canonical:** `outputs/p_z_given_u_g/20260407_183758Z/` — retain for comparison and sweep history.

**Follow-up (not blocking the freeze):** marginal **e_y_star** calibration may still be conservative in the tails (see `reports/p_z_given_u_g_native_circular_report.md` and `plots/pit_test_post_temp.png` in `185923Z`). Revisit if downstream consumers require tighter tail behavior.
