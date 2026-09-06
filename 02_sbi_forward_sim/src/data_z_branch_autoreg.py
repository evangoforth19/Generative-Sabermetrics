"""Data loading for branch_autoreg_bounded_hybrid_mdn_z stage-z."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_u import (
    build_categorical_vocabs,
    encode_categories,
    load_standardization_stats,
    merge_player_constants,
    validate_no_event_leakage,
)
from .data_z import (
    _project_root,
    _wrap_pi,
    validate_z_dataframe,
)
from .feature_contract_z import (
    load_feature_contract_z,
    validate_preprocessing_stats_z,
    validate_stage_z_config_against_contract,
    validate_vocab_sizes_z,
)
from .split_z_validation import split_z_calibration_dataframe
from .target_transforms_z import (
    BoundedLogitZscoreState,
    compute_x_bounds_per_row,
    fit_bounded_logit_zscore_state,
    jacobian_log_r_ex_std_wrt_e_x,
    jacobian_log_r_x_std_wrt_x,
    jacobian_log_r_y_std_wrt_e_y,
    raw_to_model_coords,
)


@dataclass
class ZArraysBranchAutoreg:
    x_num: np.ndarray
    cat: dict[str, np.ndarray]
    r_xy_std: np.ndarray  # (N, 2)
    r_ex_std: np.ndarray  # (N,)
    psi_rad: np.ndarray
    y_raw: np.ndarray  # (N, 4) x, psi_deg, e_y_star, e_x
    w: np.ndarray
    event_id: np.ndarray
    x_lower: np.ndarray
    x_upper: np.ndarray
    transform_state: BoundedLogitZscoreState
    j_log_raw: np.ndarray  # (N,) sum of transform Jacobians
    feature_names: list[str]


def dataframe_to_arrays_z_branch_autoreg(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    num_order: list[str],
    weight_col: str,
    transform_state: BoundedLogitZscoreState,
    *,
    train_x_for_bounds: np.ndarray | None = None,
    x_cfg: dict[str, Any],
) -> ZArraysBranchAutoreg:
    for c in num_order:
        if c not in stats:
            raise ValueError(f"No standardization stats for {c!r}")
    x_parts = []
    for c in num_order:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        mu, sig = stats[c]["mean"], stats[c]["std"]
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        x_parts.append((v - mu) / sig)
    x_num = np.stack(x_parts, axis=1).astype(np.float32)

    cat: dict[str, np.ndarray] = {}
    for col in vocabs:
        cat[col] = encode_categories(df, col, vocabs[col])

    x_raw = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
    psi_deg = pd.to_numeric(df["psi_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ey_raw = pd.to_numeric(df["e_y_star"], errors="coerce").to_numpy(dtype=np.float64)
    ex_raw = pd.to_numeric(df["e_x"], errors="coerce").to_numpy(dtype=np.float64)
    bat = pd.to_numeric(df["bat_length_in"], errors="coerce").to_numpy(dtype=np.float64)

    x_lo, x_hi, _, _ = compute_x_bounds_per_row(
        bat, cfg_x=x_cfg, train_x=train_x_for_bounds
    )
    r_x, r_y, r_ex, aux = raw_to_model_coords(
        x_raw, ey_raw, ex_raw, x_lower=x_lo, x_upper=x_hi, state=transform_state
    )
    j_log = (
        jacobian_log_r_x_std_wrt_x(x_raw, x_lo, x_hi, transform_state, s_x_clip=aux["s_x_clip"])
        + jacobian_log_r_y_std_wrt_e_y(ey_raw, transform_state, s_y_clip=aux["s_y_clip"])
        + jacobian_log_r_ex_std_wrt_e_x(ex_raw, transform_state, s_ex_clip=aux["s_ex_clip"])
    ).astype(np.float32)
    r_xy_std = np.stack([r_x, r_y], axis=1).astype(np.float32)
    psi_rad = _wrap_pi(np.radians(psi_deg)).astype(np.float32)
    y_raw = np.column_stack([x_raw, psi_deg, ey_raw, ex_raw]).astype(np.float32)

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev = df["event_id"].to_numpy(dtype=np.int64)

    if np.isnan(x_num).any() or np.isnan(r_xy_std).any():
        raise ValueError("NaN after branch-autoreg z arrays")

    return ZArraysBranchAutoreg(
        x_num=x_num,
        cat=cat,
        r_xy_std=r_xy_std,
        r_ex_std=r_ex.astype(np.float32),
        psi_rad=psi_rad,
        y_raw=y_raw,
        w=w.astype(np.float32),
        event_id=ev,
        x_lower=x_lo.astype(np.float32),
        x_upper=x_hi.astype(np.float32),
        transform_state=transform_state,
        j_log_raw=j_log,
        feature_names=list(num_order),
    )


def load_all_z_data_branch_autoreg_bounded(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
    feature_contract: dict[str, Any] | None = None,
) -> tuple[
    ZArraysBranchAutoreg,
    ZArraysBranchAutoreg,
    ZArraysBranchAutoreg,
    ZArraysBranchAutoreg,
    ZArraysBranchAutoreg,
    dict[str, Any],
]:
    """
    Returns train, val_select, temp_cal, test, bundle.

    Splits z_calibration into val_select / temp_cal when validation.split_calibration_for_model_selection.
    """
    root = _project_root(project_root)
    contract = feature_contract if feature_contract is not None else load_feature_contract_z(cfg, root)
    validate_stage_z_config_against_contract(cfg, contract)
    targets = tuple(contract["targets"])
    if targets != ("x", "psi_deg", "e_y_star", "e_x"):
        raise ValueError(f"branch autoreg expects targets (x, psi_deg, e_y_star, e_x); got {targets}")

    paths = cfg["paths"]
    stats: dict[str, dict[str, float]] = copy.deepcopy(
        load_standardization_stats(root / paths["standardization_stats"])
    )
    master = pd.read_parquet(root / paths["context_master"])

    splits: dict[str, pd.DataFrame] = {}
    for split, key in [("train", "z_train"), ("calibration", "z_calibration"), ("test", "z_test")]:
        df = pd.read_parquet(root / paths[key])
        validate_z_dataframe(df, split, contract=contract, weight_col=cfg["weight_column"])
        df = merge_player_constants(df, master)
        splits[split] = df

    ev_t = set(splits["train"]["event_id"].unique())
    ev_c = set(splits["calibration"]["event_id"].unique())
    ev_s = set(splits["test"]["event_id"].unique())
    validate_no_event_leakage(ev_t, ev_c, ev_s)

    vcfg = cfg.get("validation", {})
    val_df = splits["calibration"]
    temp_df = splits["calibration"]
    split_summary: dict[str, Any] = {}
    if vcfg.get("split_calibration_for_model_selection", False):
        val_df, temp_df, split_summary = split_z_calibration_dataframe(
            splits["calibration"],
            val_fraction=float(vcfg.get("val_select_fraction", 0.5)),
            seed=int(vcfg.get("split_seed", 20260502)),
        )
        split_summary.update(
            {
                "n_rows_train": len(splits["train"]),
                "n_rows_val_select": len(val_df),
                "n_rows_temp_cal": len(temp_df),
                "n_rows_test": len(splits["test"]),
                "n_events_train": len(ev_t),
                "n_events_val_select": len(val_df["event_id"].unique()),
                "n_events_temp_cal": len(temp_df["event_id"].unique()),
                "n_events_test": len(ev_s),
            }
        )

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        validate_vocab_sizes_z(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(splits["train"], list(cfg["categorical_features"].keys()))
        validate_vocab_sizes_z(vocabs, contract)

    num_order = list(contract["x_numeric_zscore_column_order"])
    pcols = list(contract["player_constant_features"])
    wcol = cfg["weight_column"]
    validate_preprocessing_stats_z(stats, contract)

    train_df = splits["train"]
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}

    tt_cfg = cfg.get("target_transform", {})
    x_cfg = tt_cfg.get("x", {})
    train_x = pd.to_numeric(train_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    transform_state, transform_manifest = fit_bounded_logit_zscore_state(
        train_df, target_transform_cfg=tt_cfg
    )

    train_a = dataframe_to_arrays_z_branch_autoreg(
        splits["train"],
        stats,
        vocabs,
        num_order,
        wcol,
        transform_state,
        train_x_for_bounds=train_x,
        x_cfg=x_cfg,
    )
    val_a = dataframe_to_arrays_z_branch_autoreg(
        val_df, stats, vocabs, num_order, wcol, transform_state, train_x_for_bounds=train_x, x_cfg=x_cfg
    )
    temp_a = dataframe_to_arrays_z_branch_autoreg(
        temp_df, stats, vocabs, num_order, wcol, transform_state, train_x_for_bounds=train_x, x_cfg=x_cfg
    )
    test_a = dataframe_to_arrays_z_branch_autoreg(
        splits["test"], stats, vocabs, num_order, wcol, transform_state, train_x_for_bounds=train_x, x_cfg=x_cfg
    )

    meta = {
        "n_rows_train": len(splits["train"]),
        "n_rows_val_select": len(val_df),
        "n_rows_temp_cal": len(temp_df),
        "n_rows_test": len(splits["test"]),
        "n_events_train": len(ev_t),
        "n_events_cal": len(ev_c),
        "n_events_test": len(ev_s),
        "vocabs": {k: len(v) for k, v in vocabs.items()},
        "target_parameterization": "branch_autoreg_bounded_hybrid_mdn_z",
        "model_family": "branch_autoreg_bounded_hybrid_mdn_z",
        "target_transform_manifest": transform_manifest,
        "z_targets": list(targets),
        "validation_split_summary": split_summary,
    }
    return train_a, val_a, temp_a, test_a, {"vocabs": vocabs, "meta": meta, "contract": contract, "transform_manifest": transform_manifest}
