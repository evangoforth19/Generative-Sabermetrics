"""Assemble processed SBI datasets from production artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema
from .feature_engineering import (
    add_spin_axis_trig,
    attach_release_speed_from_kinematics,
    compute_train_standardization,
    merge_statcast_release_speed,
    save_standardization_stats,
    standardization_numeric_blocks,
    z_count_from_balls_strikes,
)
from .io_utils import InputManifest, assert_columns, read_json_if_exists, read_table, ResolvedSource
from .player_constants import build_player_constants_from_selected_events
from .split_utils import attach_split_to_draws, attach_split_to_events, event_stratified_split, save_split_manifest

BIP_MODEL_QUALITY_FLAGS = [
    "flag_missing_required",
    "flag_bad_time",
    "flag_bad_vin_y",
    "flag_bad_bat_speed",
    "flag_bad_exit_speed",
    "flag_bad_launch_angle",
    "flag_bad_attack_angle",
    "flag_bad_attack_direction",
    "flag_bad_spin_rate",
    "flag_bad_spin_axis",
    "flag_bad_omega_minus",
]

def bip_model_training_event_mask(df: pd.DataFrame) -> pd.Series:
    """
    Match ``run_mcmc_posterior_bank`` bip_model row filter + fair wedge (USE_FAIR_ONLY).
    """
    need = BIP_MODEL_QUALITY_FLAGS + ["flag_outside_fair_wedge"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            "context master is missing bip_model QC / fair-wedge flags: "
            f"{missing}. Expected on aggregates/context_event_table.parquet."
        )
    m = pd.Series(True, index=df.index)
    for c in BIP_MODEL_QUALITY_FLAGS:
        m &= ~df[c].fillna(True).astype(bool)
    m &= ~df["flag_outside_fair_wedge"].fillna(True).astype(bool)
    return m


def discover_production_inputs(production_root: Path, manifest: InputManifest) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    specs: list[tuple[str, str, str | None]] = [
        ("selected_events", "selected_events.parquet", "selected_events.csv"),
        ("context_event_table", "aggregates/context_event_table.parquet", None),
        ("event_quality_flags", "aggregates/event_quality_flags.parquet", None),
        ("chain_diagnostics", "aggregates/chain_diagnostics.parquet", None),
        ("posterior_draws_long_raw", "train_exports/posterior_draws_long_raw.parquet", None),
        ("posterior_draws_long_basic", "train_exports/posterior_draws_long_basic_screened.parquet", None),
        ("posterior_draws_long_strict", "train_exports/posterior_draws_long_strict_screened.parquet", None),
        ("context_to_latent", "train_exports/context_to_latent_train.parquet", "train_exports/context_to_latent_train.csv"),
        ("context_to_upstream", "train_exports/context_to_upstream_train.parquet", "train_exports/context_to_upstream_train.csv"),
        ("context_upstream_latent_joint", "train_exports/context_upstream_latent_joint_train.parquet", "train_exports/context_upstream_latent_joint_train.csv"),
        ("event_level_targets", "train_exports/event_level_targets.parquet", None),
        ("feature_schema", "train_exports/feature_schema.json", None),
        ("preprocessing_artifacts", "train_exports/preprocessing_artifacts.json", None),
        ("split_manifest_production", "train_exports/split_manifest.json", None),
        ("observed_inference_pack", "calibration_exports/observed_inference_pack.parquet", None),
    ]
    for key, pq_rel, csv_rel in specs:
        pq = production_root / pq_rel
        if pq.is_file():
            manifest.sources.append(ResolvedSource(key, str(pq.resolve()), "parquet", True, ""))
            out[key] = pq
            continue
        if csv_rel:
            csv = production_root / csv_rel
            if csv.is_file():
                manifest.sources.append(
                    ResolvedSource(key, str(csv.resolve()), "csv", True, "parquet missing; CSV fallback")
                )
                out[key] = csv
                continue
        manifest.sources.append(ResolvedSource(key, str(pq.resolve()), "missing", False, ""))
        out[key] = None
    return out


def build_context_event_master(
    paths: dict[str, Path | None],
    manifest_extra: dict[str, Any],
    statcast_pickle: Path | None,
) -> pd.DataFrame:
    ctx_p = paths["context_event_table"]
    q_p = paths["event_quality_flags"]
    ch_p = paths["chain_diagnostics"]
    if ctx_p is None:
        raise FileNotFoundError("context_event_table missing")
    ctx = read_table(ctx_p)
    assert_columns(ctx, ["event_id", "batter_name"], "context_event_table")

    ctx = z_count_from_balls_strikes(ctx, table_name="context_event_table")
    if "release_speed" in ctx.columns:
        manifest_extra["release_speed_source"] = "context_event_table"
    elif statcast_pickle is not None and statcast_pickle.is_file():
        ctx = merge_statcast_release_speed(ctx, statcast_pickle, table_name="context_event_table")
        manifest_extra["release_speed_source"] = f"statcast_pickle:{statcast_pickle.resolve()}"
    else:
        # No room for full Statcast pickle: ‖(vx0,vy0,vz0)‖ ft/s → mph (same convention as SBI kinematics block).
        ctx = attach_release_speed_from_kinematics(ctx, table_name="context_event_table")
        manifest_extra["release_speed_source"] = (
            "derived_sqrt_vx0_vy0_vz0_ft_s_to_mph_no_statcast_pickle"
        )
    ctx = add_spin_axis_trig(ctx, "spin_axis")

    if q_p is not None:
        q = read_table(q_p)
        ctx = ctx.merge(q, on=["event_id", "batter_name"], how="left", suffixes=("", "_qualdup"))
    if ch_p is not None:
        ch = read_table(ch_p)
        dup = [c for c in ch.columns if c in ctx.columns and c not in ("event_id", "batter_name")]
        ch = ch.drop(columns=[c for c in dup if c in ch.columns], errors="ignore")
        ctx = ctx.merge(ch, on=["event_id", "batter_name"], how="left", suffixes=("", "_chaindup"))

    if "launch_speed" in ctx.columns:
        ctx["exit_speed_obs_mph"] = pd.to_numeric(ctx["launch_speed"], errors="coerce")
    if "launch_angle" in ctx.columns:
        ctx["launch_angle_obs_deg"] = pd.to_numeric(ctx["launch_angle"], errors="coerce")
    if "spray_angle_deg" in ctx.columns:
        ctx["spray_angle_obs_deg"] = pd.to_numeric(ctx["spray_angle_deg"], errors="coerce")

    return ctx


def _forbid_phi_as_theta(draws: pd.DataFrame, label: str) -> None:
    """Reject ambiguous plain ``theta`` without ``_deg`` disambiguation."""
    if "theta" in draws.columns:
        raise ValueError(
            f"{label}: ambiguous column `theta`; production must not use unqualified theta "
            f"(canonical stage-z launch angle is `theta_deg`)."
        )


def _build_canonical_theta_deg(
    df: pd.DataFrame,
    label: str,
    manifest_extra: dict[str, Any],
) -> None:
    """
    Set ``theta_deg`` per draw: finite ``theta_star_deg`` else ``theta_obs_deg`` from launch.

    Keeps ``theta_obs_deg`` and optionally ``theta_star_deg`` on full draw tables for lineage;
    stage-z slices use ``theta_deg`` only.
    Never uses ``phi_star`` for launch angle.
    """
    has_lobs = "launch_angle_obs_deg" in df.columns
    if not has_lobs:
        raise ValueError(
            f"{label}: missing `launch_angle_obs_deg` on draws — cannot build `theta_obs_deg` / `theta_deg`."
        )
    theta_obs = pd.to_numeric(df["launch_angle_obs_deg"], errors="coerce").astype(np.float64)
    df["theta_obs_deg"] = theta_obs

    if "theta_star_deg" in df.columns:
        star = pd.to_numeric(df["theta_star_deg"], errors="coerce").astype(np.float64)
        df["theta_star_deg"] = star
    else:
        star = pd.Series(np.nan, index=df.index, dtype=np.float64)

    use_star = np.isfinite(star.to_numpy())
    obs_a = theta_obs.to_numpy()
    star_a = star.to_numpy()
    theta_vals = np.where(use_star, star_a, obs_a)
    if not np.all(np.isfinite(theta_vals)):
        n_bad = int(np.sum(~np.isfinite(theta_vals)))
        raise ValueError(
            f"{label}: {n_bad} rows have non-finite `theta_deg` after coalesce "
            f"(need finite `theta_star_deg` or `launch_angle_obs_deg`)."
        )
    df["theta_deg"] = theta_vals

    n_star = int(use_star.sum())
    n_obs = int(len(df) - n_star)
    manifest_extra[f"z_theta_row_stats_{label}"] = {
        "canonical_column": "theta_deg",
        "n_rows_from_theta_star_deg": n_star,
        "n_rows_from_theta_obs_deg": n_obs,
    }
    manifest_extra[f"z_theta_provenance_{label}"] = (
        "Per draw: `theta_deg` = finite `theta_star_deg` if available, else `theta_obs_deg` from "
        "`launch_angle_obs_deg`; never `phi_star`."
    )


def _prepare_draw_table(
    path: Path | None,
    master: pd.DataFrame,
    player_consts: pd.DataFrame,
    manifest_extra: dict[str, Any],
    label: str,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = read_table(path)
    assert_columns(df, ["event_id", "batter_name"], label)

    m = master.copy()
    m["batter_key"] = m["batter_name"].astype(str).str.strip().str.lower()
    df = df.copy()
    df["batter_key"] = df["batter_name"].astype(str).str.strip().str.lower()
    merge_cols = ["event_id"] + [
        c for c in m.columns if c not in ("batter_key", "event_id") and c not in df.columns
    ]
    if len(merge_cols) > 1:
        df = df.merge(m[merge_cols], on="event_id", how="left")

    pc = player_consts.copy()
    pc["batter_key"] = pc["batter_name"].astype(str).str.strip().str.lower()
    pc = pc.drop(columns=["batter_name"])
    df = df.merge(pc, on="batter_key", how="left")
    df.drop(columns=["batter_key"], inplace=True, errors="ignore")

    for c in schema.U_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"{label}: missing required stage-u column {c!r}")

    _build_canonical_theta_deg(df, label, manifest_extra)
    _forbid_phi_as_theta(df, label)

    if "psi_deg" not in df.columns and "psi" in df.columns:
        df["psi_deg"] = pd.to_numeric(df["psi"], errors="coerce")
    elif "psi_deg" in df.columns:
        df["psi_deg"] = pd.to_numeric(df["psi_deg"], errors="coerce")

    for c in ("e_y_star", "x"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def run_full_build(
    production_root: Path,
    project_root: Path,
    split_seed: int = 42,
    statcast_pickle: Path | None = None,
) -> dict[str, Any]:
    manifest = InputManifest(production_root=str(production_root.resolve()))
    manifest_extra: dict[str, Any] = {}
    paths = discover_production_inputs(production_root, manifest)

    for req_key in (
        "posterior_draws_long_raw",
        "posterior_draws_long_basic",
        "posterior_draws_long_strict",
    ):
        if paths[req_key] is None:
            raise FileNotFoundError(
                f"Missing required production export for key {req_key!r} under {production_root}"
            )

    sel_p = paths["selected_events"]
    if sel_p is None:
        raise FileNotFoundError("selected_events missing")
    selected = read_table(sel_p)

    _mmc2_root = project_root.resolve().parent
    pickle_path: Path | None = None
    if statcast_pickle is not None and statcast_pickle.is_file():
        pickle_path = statcast_pickle
    else:
        for name in ("batter_data_2020_2025.pkl", "batter_data_2024_2025.pkl"):
            cand = _mmc2_root / name
            if cand.is_file():
                pickle_path = cand
                break

    master_full = build_context_event_master(paths, manifest_extra, pickle_path)
    manifest_extra["context_rows_before_bip_filter"] = int(len(master_full))
    elig = bip_model_training_event_mask(master_full)
    manifest_extra["context_rows_after_bip_filter"] = int(elig.sum())
    master = master_full.loc[elig].copy()
    if master.empty:
        raise ValueError("No events left after bip_model quality + inside fair-wedge filter.")

    player_consts = build_player_constants_from_selected_events(
        selected.loc[selected["event_id"].isin(master["event_id"])].copy()
    )
    mk = master.copy()
    pk = player_consts.copy()
    pk["batter_key"] = pk["batter_name"].astype(str).str.strip().str.lower()
    mk["batter_key"] = mk["batter_name"].astype(str).str.strip().str.lower()
    pk_drop = pk.drop(columns=["batter_name"])
    for c in schema.P_COLUMNS:
        if c in mk.columns:
            mk = mk.drop(columns=[c])
    master = mk.merge(pk_drop, on="batter_key", how="left").drop(columns=["batter_key"])

    proc = project_root / "data_processed"
    inter = project_root / "data_intermediate"
    man_dir = project_root / "manifests"
    rep_dir = project_root / "reports"
    for d in (proc, project_root / "data_raw", inter, man_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    player_consts.to_parquet(proc / "sbi_player_constants.parquet", index=False)
    master.to_parquet(proc / "sbi_context_event_master.parquet", index=False)

    def _filter_draws_to_master(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df.empty:
            return df
        allowed = set(master["event_id"].tolist())
        out = df.loc[df["event_id"].isin(allowed)].copy()
        manifest_extra[f"draw_rows_{name}_after_bip_event_filter"] = int(len(out))
        return out

    draws_raw = _prepare_draw_table(paths["posterior_draws_long_raw"], master, player_consts, manifest_extra, "raw")
    draws_basic = _prepare_draw_table(paths["posterior_draws_long_basic"], master, player_consts, manifest_extra, "basic")
    draws_strict = _prepare_draw_table(
        paths["posterior_draws_long_strict"], master, player_consts, manifest_extra, "strict"
    )
    if draws_strict.empty:
        raise ValueError("Strict screened posterior draws table is empty after filters.")

    draws_raw = _filter_draws_to_master(draws_raw, "raw")
    draws_basic = _filter_draws_to_master(draws_basic, "basic")
    draws_strict = _filter_draws_to_master(draws_strict, "strict")

    manifest_extra["draw_rows_raw_final"] = int(len(draws_raw))
    manifest_extra["draw_rows_basic_final"] = int(len(draws_basic))
    manifest_extra["draw_rows_strict_final"] = int(len(draws_strict))

    for _lab, _df in ("raw", draws_raw), ("basic", draws_basic), ("strict", draws_strict):
        if _df.empty:
            continue
        if "theta_deg" not in _df.columns:
            raise ValueError(f"Draw table {_lab} missing canonical `theta_deg` after preprocessing.")

    manifest_extra["z_theta_launch_column"] = "theta_deg"
    manifest_extra["z_theta_provenance"] = manifest_extra.get(
        "z_theta_provenance_basic",
        manifest_extra.get("z_theta_provenance_raw", ""),
    )
    manifest_extra["z_theta_row_stats"] = {
        "basic": manifest_extra.get("z_theta_row_stats_basic", {}),
        "strict": manifest_extra.get("z_theta_row_stats_strict", {}),
        "raw": manifest_extra.get("z_theta_row_stats_raw", {}),
    }

    if len(draws_basic) == 0 or len(draws_strict) == 0:
        raise ValueError(
            "No posterior draws remain after bip_model event filtering. "
            "Check alignment between context_event_table and train_exports draw tables."
        )

    draws_raw.to_parquet(proc / "sbi_draws_raw.parquet", index=False)
    draws_basic.to_parquet(proc / "sbi_draws_basic.parquet", index=False)
    draws_strict.to_parquet(proc / "sbi_draws_strict.parquet", index=False)

    ev_table = master[["event_id", "batter_name"]].drop_duplicates()
    split_df = event_stratified_split(
        ev_table["event_id"].to_numpy(),
        ev_table["batter_name"].to_numpy(),
        seed=split_seed,
    )
    save_split_manifest(split_df, man_dir / "split_manifest.json", 0.7, 0.15, 0.15, split_seed)
    split_df.to_parquet(proc / "event_split_table.parquet", index=False)

    draws_basic_s = attach_split_to_draws(draws_basic, split_df)
    draws_strict_s = attach_split_to_draws(draws_strict, split_df)

    train_mask = draws_basic_s["split"].eq("train")
    num_cols = standardization_numeric_blocks(draws_basic_s)
    stats = compute_train_standardization(draws_basic_s.loc[train_mask], num_cols)
    save_standardization_stats(stats, man_dir / "standardization_stats.json")

    gcols = [c for c in schema.G_COLUMNS if c in draws_basic_s.columns]
    pcols = [c for c in schema.P_COLUMNS if c in draws_basic_s.columns]
    ucols = [c for c in schema.U_COLUMNS if c in draws_basic_s.columns]
    zcols = [c for c in schema.Z_TARGET_COLUMNS if c in draws_basic_s.columns]
    # Posterior exports may include tangential restitution e_x for extended stage-z (theta-free) training.
    if "e_x" in draws_basic_s.columns and "e_x" not in zcols:
        zcols.append("e_x")

    weight_cols = [
        c
        for c in [
            "uniform_draw_weight_within_event",
            "normalized_log_target_weight",
            "event_quality_weight",
            "combined_training_weight",
        ]
        if c in draws_basic_s.columns
    ]
    qual_cols = [
        c
        for c in [
            "posterior_reliability_class",
            "event_quality_score",
            "event_reliability_weight",
            "poor_mixing_flag",
            "x_near_support_flag",
            "psi_near_support_flag",
            "e_y_near_bound_flag",
        ]
        if c in draws_basic_s.columns
    ]
    id_cols = ["event_id", "batter_name", "split"]

    def chunk_u(df: pd.DataFrame) -> pd.DataFrame:
        keep = id_cols + gcols + pcols + ucols + weight_cols + qual_cols
        keep = [c for c in keep if c in df.columns]
        return df[keep].copy()

    # Stage-z primary: basic-screened draws (same event split as u_*) for trainable support.
    # Strict-screened draws archived separately for analysis (not sole training source).
    z_use = draws_basic_s.copy()
    manifest_extra["z_screen_choice"] = "basic_screened_primary"
    manifest_extra["z_strict_archival"] = "z_strict_{train,calibration,test}.parquet from draws_strict_s"

    def chunk_z(df: pd.DataFrame) -> pd.DataFrame:
        keep = id_cols + gcols + pcols + ucols + zcols + weight_cols + qual_cols
        keep = [c for c in keep if c in df.columns]
        return df[keep].copy()

    for part_name, part_key in [("train", "train"), ("calibration", "calibration"), ("test", "test")]:
        mask_b = draws_basic_s["split"].eq(part_key)
        chunk_u(draws_basic_s.loc[mask_b]).to_parquet(proc / f"u_{part_name}.parquet", index=False)
        mask_z = z_use["split"].eq(part_key)
        chunk_z(z_use.loc[mask_z]).to_parquet(proc / f"z_{part_name}.parquet", index=False)
        # Optional strict-only slice (same schema) for debugging / strict-aligned experiments
        mask_sz = draws_strict_s["split"].eq(part_key)
        sub_s = draws_strict_s.loc[mask_sz]
        if len(sub_s) == 0:
            chunk_z(z_use.iloc[:0]).to_parquet(proc / f"z_strict_{part_name}.parquet", index=False)
        else:
            chunk_z(sub_s).to_parquet(proc / f"z_strict_{part_name}.parquet", index=False)

    master_y = master.set_index("event_id")
    z_joint = z_use.copy()
    z_joint["EV"] = z_joint["event_id"].map(master_y["exit_speed_obs_mph"])
    z_joint["LA"] = z_joint["event_id"].map(master_y["launch_angle_obs_deg"])
    z_joint["SA"] = z_joint["event_id"].map(master_y["spray_angle_obs_deg"])

    zcol_set = set(zcols)
    joint_extra_latent = [
        c for c in ("e_x", "omega_plus", "raw_e_x") if c in z_joint.columns and c not in zcol_set
    ]
    joint_theta_lineage = [c for c in ("theta_star_deg", "theta_obs_deg") if c in z_joint.columns]
    joint_keep = (
        id_cols
        + gcols
        + pcols
        + ucols
        + zcols
        + joint_theta_lineage
        + joint_extra_latent
        + ["EV", "LA", "SA"]
        + weight_cols
        + qual_cols
    )
    joint_keep = [c for c in joint_keep if c in z_joint.columns]
    joint_keep = list(dict.fromkeys(joint_keep))
    for part_name, part_key in [("train", "train"), ("calibration", "calibration"), ("test", "test")]:
        mask = z_joint["split"].eq(part_key)
        z_joint.loc[mask, joint_keep].to_parquet(proc / f"joint_{part_name}.parquet", index=False)

    gpm = attach_split_to_events(master, split_df)
    base_cols = ["event_id", "batter_name", "split"]
    pcols_b = [c for c in schema.P_COLUMNS if c in gpm.columns]
    base_keep = base_cols + gcols + pcols_b + qual_cols
    base_keep = [c for c in base_keep if c in gpm.columns]
    gpm["EV"] = pd.to_numeric(gpm["exit_speed_obs_mph"], errors="coerce")
    gpm["LA"] = pd.to_numeric(gpm["launch_angle_obs_deg"], errors="coerce")
    gpm["SA"] = pd.to_numeric(gpm["spray_angle_obs_deg"], errors="coerce")
    for part_name, part_key in [("train", "train"), ("calibration", "calibration"), ("test", "test")]:
        sub = gpm.loc[gpm["split"].eq(part_key), base_keep + ["EV", "LA", "SA"]].drop_duplicates("event_id")
        sub.to_parquet(proc / f"baseline_direct_y_{part_name}.parquet", index=False)

    manifest.extra = {
        **manifest_extra,
        "g_columns_resolved": gcols,
        "p_columns_resolved": pcols,
        "u_columns_resolved": ucols,
        "z_columns_resolved": zcols,
        "z_theta_launch_column": manifest_extra.get("z_theta_launch_column"),
        "z_theta_provenance": manifest_extra.get("z_theta_provenance"),
        "z_theta_resolution": {
            k: v for k, v in manifest_extra.items() if k.startswith("z_theta_launch_column_")
        },
        "feature_schema_production": read_json_if_exists(production_root / "train_exports/feature_schema.json"),
        "split_manifest_production": read_json_if_exists(production_root / "train_exports/split_manifest.json"),
    }
    manifest.save(man_dir / "input_discovery.json")

    theta_note = str(manifest_extra.get("z_theta_provenance", ""))
    (man_dir / "modeling_schema.json").write_text(
        json.dumps(
            schema.modeling_schema_manifest(
                manifest_extra.get("z_theta_provenance", ""),
                z_theta_stats=manifest_extra.get("z_theta_row_stats"),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    audit_lines = [
        "# Dataset audit",
        "",
        f"production_root: `{production_root}`",
        "",
        "## Discovered inputs",
        "\n".join(
            f"- `{s.logical_name}`: {s.format} `{s.resolved_path}` {'(' + s.note + ')' if s.note else ''}"
            for s in manifest.sources
            if s.exists
        ),
        "",
        "## Row counts",
        f"- sbi_draws_raw: {len(draws_raw):,}",
        f"- sbi_draws_basic: {len(draws_basic):,}",
        f"- sbi_draws_strict: {len(draws_strict):,}",
        f"- context master rows: {len(master):,} events unique: {master['event_id'].nunique()}",
        "",
        "## Split counts",
        split_df["split"].value_counts().to_string(),
        "",
        "## Z stage",
        manifest_extra.get("z_screen_choice", "strict_only"),
        "",
        "## Stage-z launch angle (SBI canonical)",
        f"- canonical column on processed **z** slices: `{manifest_extra.get('z_theta_launch_column')}`",
        f"- provenance: {manifest_extra.get('z_theta_provenance', '')}",
        "",
        "- **Broader production** long draws may still carry `e_x`, `omega_plus`, etc.; "
        "those are **not** SBI stage-z targets (decoder nuisances).",
        "",
        json.dumps(manifest_extra.get("z_theta_row_stats", {}), indent=2),
        "",
        "## Resolved g columns",
        ", ".join(gcols),
        "",
    ]
    (rep_dir / "DATASET_AUDIT.md").write_text("\n".join(audit_lines), encoding="utf-8")

    from .audit_reports import write_all_audit_reports

    write_all_audit_reports(production_root, project_root, manifest_extra)

    return {
        "processed_dir": str(proc),
        "manifest_dir": str(man_dir),
        "theta_note": theta_note,
        "z_screen": manifest_extra.get("z_screen_choice", "strict_only"),
    }
