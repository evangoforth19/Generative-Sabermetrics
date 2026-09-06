# SBI forward simulation — data foundation

This package prepares training and calibration datasets for a **two-stage** neural simulator:

1. **Stage A:** \(p(u \mid g, p)\) with \(u = (v_{ss,\tilde{}}, a_{\tilde{}}, d_{\tilde{}})\).
2. **Stage B:** \(p(z \mid u, g, p)\) with **reduced** \(z = (x, \psi_{\mathrm{deg}}, e_y^\star, \theta_{\mathrm{deg}})\).  
   Column names: `x`, `psi_deg`, `e_y_star`, `theta_deg`. Here **`theta_deg`** means finite **`theta_star_deg`** when available on the draw, else observed launch from **`launch_angle_obs_deg`** (never `phi_star`).  
   **`e_x`** and **`omega_plus`** are *not* stage-z targets; they stay on joint/raw production tables and are handled in the **deterministic decoder** (see `src/physics_decoder_contract.py`).
3. **Decoder:** deterministic physics mapping \((u,z,g,p) \to (EV, LA, SA)\), solving nuisances such as `e_x`, `omega_plus` under production-aligned admissibility (`src/physics_decoder.py`).  
   **Context `g` is stricter than neural `g`:** stage-u / stage-z models use `schema.G_COLUMNS` only; `decode_bip_from_sample` requires the locked set in `manifests/physics_decoder_input_contract.json` (notably **`release_pos_y`** for bank contact-time parity). Enforcement: `validate_decoder_inputs` → `DecoderInputError` on violation.

## Layout

- `configs/` — reserved for training configs (future).
- `data_raw/` — reserved; inputs are read from production `Outputs/.../production/`.
- `data_intermediate/` — scratch (future).
- `data_processed/` — outputs from the build script (parquet tables).
- `manifests/` — `input_discovery.json`, `split_manifest.json`, `standardization_stats.json`, `modeling_schema.json`, `physics_decoder_input_contract.json`.
- `reports/` — `DATASET_AUDIT.md`.
- `scripts/` — runnable entrypoints.
- `src/` — library code.

## Run preprocessing

From `MCMC 2/` (with `.venv_prod` activated):

```bash
python sbi_forward_sim/scripts/build_sbi_datasets.py \
  --production-root "Outputs/refactor_exact_root_rhh/production" \
  --project-root "sbi_forward_sim"
```

Validate:

```bash
python sbi_forward_sim/scripts/validate_sbi_datasets.py --project-root "sbi_forward_sim"
```

## Outputs (processed)

| File | Role |
|------|------|
| `sbi_player_constants.parquet` | One row per batter: bat / COM / inertia |
| `sbi_context_event_master.parquet` | Event-level context, quality, chain diagnostics, observed BIP |
| `sbi_draws_{raw,basic,strict}.parquet` | Draw-level tables merged with context + player constants + weights |
| `event_split_table.parquet` | `event_id` → train / calibration / test |
| `u_{train,calibration,test}.parquet` | Stage A training packs (basic-screened draws) |
| `z_{train,calibration,test}.parquet` | Stage B packs (strict when enough data) |
| `joint_*.parquet` | Debug: g, p, u, z, y together |
| `baseline_direct_y_*.parquet` | Direct \(y\) prediction baseline (only g, p, y) |

No model training is performed in this step.

## Provisional canonical checkpoints (trained models)

Default trained runs for the forward stack (see `OPEN_DECISIONS.md` for lineage notes):

- **Stage-u:** `outputs/p_u_given_g/20260407_173057Z/`
- **Stage-z:** `outputs/p_z_given_u_g/20260407_185923Z/` (hybrid **native** circular likelihood: 2D full-cov Gaussian on `(x, e_y_star)` + von Mises on `psi` / `theta`, shared mixture weights)

Prior **6D sin/cos Gaussian** stage-z reference (not canonical): `outputs/p_z_given_u_g/20260407_183758Z/`.
