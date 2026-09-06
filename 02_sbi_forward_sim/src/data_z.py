"""Load and tensorize stage-z datasets p(z|u,g,p)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_u import (
    build_categorical_vocabs,
    encode_categories,
    index_by_event,
    load_standardization_stats,
    merge_player_constants,
    validate_no_event_leakage,
)
from .feature_contract_z import (
    load_feature_contract_z,
    validate_preprocessing_stats_z,
    validate_stage_z_config_against_contract,
    validate_vocab_sizes_z,
)
from .schema import Z_TARGET_COLUMNS
from .target_transform_z import (
    CIRCULAR_SIX_ORDER,
    circular_target_transform_manifest,
    raw_four_to_circ_six,
)


def _project_root(project_root: Path | None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def validate_z_dataframe(
    df: pd.DataFrame,
    split_name: str,
    *,
    contract: dict[str, Any],
    require_weight: bool = True,
    weight_col: str = "combined_training_weight",
) -> None:
    targets = list(contract["targets"]) if contract.get("targets") else list(Z_TARGET_COLUMNS)
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"{split_name}: missing stage-z target {t!r} (from feature contract)")
    for c in contract["upstream_u_features"]:
        if c not in df.columns:
            raise ValueError(f"{split_name}: missing upstream u column {c!r}")
    for c in contract["numeric_context_features"]:
        if c not in df.columns:
            raise ValueError(f"{split_name}: missing numeric context {c!r}")
    for c in contract["categorical_yaml_key_order"]:
        if c not in df.columns:
            raise ValueError(f"{split_name}: missing categorical {c!r}")
    if contract.get("phi_star_forbidden") and "phi_star" in df.columns:
        raise ValueError(f"{split_name}: phi_star must not appear on stage-z table")
    for bad in contract.get("forbidden_neural_input_columns", []):
        if bad in df.columns:
            # allowed on disk, but must not be in model cfg — checked elsewhere
            pass
    if require_weight and weight_col not in df.columns:
        raise ValueError(f"{split_name}: missing weight {weight_col!r}")


@dataclass
class ZArrays:
    x_num: np.ndarray
    cat: dict[str, np.ndarray]
    y: np.ndarray
    y_raw: np.ndarray
    w: np.ndarray
    event_id: np.ndarray
    target_means: np.ndarray
    target_stds: np.ndarray
    feature_names: list[str]


@dataclass
class ZArraysCircular:
    """6D standardized circular parameterization; y_raw_four keeps raw contract semantics."""

    x_num: np.ndarray
    cat: dict[str, np.ndarray]
    y: np.ndarray  # (N, 6) standardized internal targets
    y_raw_four: np.ndarray  # (N, 4) x, psi_deg, e_y_star, theta_deg
    w: np.ndarray
    event_id: np.ndarray
    circ_means: np.ndarray
    circ_stds: np.ndarray
    circ_names: list[str]
    feature_names: list[str]


@dataclass
class ZArraysNativeCircular:
    """
    Hybrid native-circular targets: standardized (x, e_y_star) + radians (psi, theta).

    Angles are in [-pi, pi); raw four-vector preserves contract column order for evaluation.
    """

    x_num: np.ndarray
    cat: dict[str, np.ndarray]
    y_gauss: np.ndarray  # (N, 2) standardized x, e_y_star
    psi_rad: np.ndarray  # (N,)
    theta_rad: np.ndarray  # (N,)
    y_raw_four: np.ndarray  # (N, 4) x, psi_deg, e_y_star, theta_deg
    w: np.ndarray
    event_id: np.ndarray
    gauss_means: np.ndarray  # (2,) for x, e_y_star
    gauss_stds: np.ndarray
    feature_names: list[str]


@dataclass
class ZArraysNativeTruncEx:
    """Stage-z with (x, e_y*) Gaussian block, von Mises(psi), and raw ``e_x`` for trunc-normal training."""

    x_num: np.ndarray
    cat: dict[str, np.ndarray]
    y_gauss: np.ndarray  # (N, 2) standardized x, e_y_star
    psi_rad: np.ndarray
    e_x: np.ndarray  # (N,) raw tangential restitution in [0, 0.6] scale
    y_raw_targets: np.ndarray  # (N, 4) x, psi_deg, e_y_star, e_x
    w: np.ndarray
    event_id: np.ndarray
    gauss_means: np.ndarray
    gauss_stds: np.ndarray
    feature_names: list[str]


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return np.remainder(a + np.pi, 2.0 * np.pi) - np.pi


def dataframe_to_arrays_z(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    num_order: list[str],
    weight_col: str,
) -> ZArrays:
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

    targets = list(Z_TARGET_COLUMNS)
    y_raw = np.stack(
        [pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64) for t in targets],
        axis=1,
    )
    t_means = np.array([stats[t]["mean"] for t in targets], dtype=np.float64)
    t_stds = np.array([max(stats[t]["std"], 1e-8) for t in targets], dtype=np.float64)
    y = (y_raw - t_means) / t_stds

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev = df["event_id"].to_numpy(dtype=np.int64)

    if np.isnan(x_num).any() or np.isnan(y).any():
        raise ValueError("NaN after standardizing z arrays")

    return ZArrays(
        x_num=x_num,
        cat=cat,
        y=y.astype(np.float32),
        y_raw=y_raw.astype(np.float32),
        w=w.astype(np.float32),
        event_id=ev,
        target_means=t_means,
        target_stds=t_stds,
        feature_names=list(num_order),
    )


def load_all_z_data(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
) -> tuple[ZArrays, ZArrays, ZArrays, dict[str, Any]]:
    root = _project_root(project_root)
    contract = load_feature_contract_z(cfg, root)
    validate_stage_z_config_against_contract(cfg, contract)

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

    split_table_path = root / paths.get("event_split_table", "data_processed/event_split_table.parquet")
    if split_table_path.is_file():
        spl = pd.read_parquet(split_table_path)
        for name, evset, key in [
            ("train", ev_t, "train"),
            ("calibration", ev_c, "calibration"),
            ("test", ev_s, "test"),
        ]:
            allowed = set(spl.loc[spl["split"].eq(key), "event_id"].unique())
            if not evset.issubset(allowed):
                bad = len(evset - allowed)
                if bad > 0:
                    raise ValueError(f"{name}: {bad} events not in official split table")

    # Target columns may be absent from an older standardization_stats.json (e.g. before theta_deg);
    # fill train-only mean/std from merged train split.
    stats_targets_added: list[str] = []
    train_df_for_tgt = splits["train"]
    for t in contract["targets"]:
        if t not in stats:
            v = pd.to_numeric(train_df_for_tgt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
            stats_targets_added.append(str(t))

    validate_preprocessing_stats_z(stats, contract)

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        for k in contract["categorical_features"]:
            if k not in vocabs:
                raise ValueError(f"vocabs_override missing {k!r}")
        validate_vocab_sizes_z(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(splits["train"], list(cfg["categorical_features"].keys()))
        validate_vocab_sizes_z(vocabs, contract)

    num_order = list(contract["x_numeric_zscore_column_order"])
    pcols = list(contract["player_constant_features"])
    wcol = cfg["weight_column"]

    train_df = splits["train"]
    p_stats_added: list[str] = []
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
            p_stats_added.append(c)

    train_a = dataframe_to_arrays_z(splits["train"], stats, vocabs, num_order, wcol)
    cal_a = dataframe_to_arrays_z(splits["calibration"], stats, vocabs, num_order, wcol)
    test_a = dataframe_to_arrays_z(splits["test"], stats, vocabs, num_order, wcol)

    meta = {
        "n_rows_train": len(splits["train"]),
        "n_rows_cal": len(splits["calibration"]),
        "n_rows_test": len(splits["test"]),
        "n_events_train": len(ev_t),
        "n_events_cal": len(ev_c),
        "n_events_test": len(ev_s),
        "vocabs": {k: len(v) for k, v in vocabs.items()},
        "player_constant_stats_from_train_only": p_stats_added,
        "target_stats_filled_train_only": stats_targets_added,
    }
    return train_a, cal_a, test_a, {"vocabs": vocabs, "meta": meta, "contract": contract}


def _build_circ_standardization(train_df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Train-only std for sin/cos; x and e_y_star use JSON stats."""
    targets = list(Z_TARGET_COLUMNS)
    y_raw = np.stack(
        [pd.to_numeric(train_df[t], errors="coerce").to_numpy(dtype=np.float64) for t in targets],
        axis=1,
    )
    circ = raw_four_to_circ_six(y_raw)
    cm = np.zeros(6, dtype=np.float64)
    cs = np.ones(6, dtype=np.float64)
    for key, j in [("x", 0), ("e_y_star", 1)]:
        if key not in stats:
            raise ValueError(f"circular transform requires {key!r} in standardization stats")
        cm[j] = stats[key]["mean"]
        cs[j] = max(float(stats[key]["std"]), 1e-8)
    for j in range(2, 6):
        v = circ[:, j]
        sig = float(np.std(v))
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        cs[j] = sig
        cm[j] = float(np.mean(v))
    return cm, cs


def dataframe_to_arrays_z_circular(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    num_order: list[str],
    weight_col: str,
    circ_means: np.ndarray,
    circ_stds: np.ndarray,
) -> ZArraysCircular:
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

    targets = list(Z_TARGET_COLUMNS)
    y_raw_four = np.stack(
        [pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64) for t in targets],
        axis=1,
    )
    y_circ = raw_four_to_circ_six(y_raw_four)
    y = (y_circ - circ_means) / circ_stds

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev = df["event_id"].to_numpy(dtype=np.int64)

    if np.isnan(x_num).any() or np.isnan(y).any():
        raise ValueError("NaN after circular z arrays")

    return ZArraysCircular(
        x_num=x_num,
        cat=cat,
        y=y.astype(np.float32),
        y_raw_four=y_raw_four.astype(np.float32),
        w=w.astype(np.float32),
        event_id=ev,
        circ_means=circ_means,
        circ_stds=circ_stds,
        circ_names=list(CIRCULAR_SIX_ORDER),
        feature_names=list(num_order),
    )


def load_all_z_data_circular(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
) -> tuple[ZArraysCircular, ZArraysCircular, ZArraysCircular, dict[str, Any]]:
    """
    Same splits and inputs as load_all_z_data; targets are 6D circular standardization.
    Raw contract targets (x, psi_deg, e_y_star, theta_deg) unchanged on disk.
    """
    root = _project_root(project_root)
    contract = load_feature_contract_z(cfg, root)
    validate_stage_z_config_against_contract(cfg, contract)

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

    split_table_path = root / paths.get("event_split_table", "data_processed/event_split_table.parquet")
    if split_table_path.is_file():
        spl = pd.read_parquet(split_table_path)
        for name, evset, key in [
            ("train", ev_t, "train"),
            ("calibration", ev_c, "calibration"),
            ("test", ev_s, "test"),
        ]:
            allowed = set(spl.loc[spl["split"].eq(key), "event_id"].unique())
            if not evset.issubset(allowed):
                bad = len(evset - allowed)
                if bad > 0:
                    raise ValueError(f"{name}: {bad} events not in official split table")

    stats_targets_added: list[str] = []
    train_df_for_tgt = splits["train"]
    for t in contract["targets"]:
        if t not in stats:
            v = pd.to_numeric(train_df_for_tgt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
            stats_targets_added.append(str(t))

    validate_preprocessing_stats_z(stats, contract)

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        for k in contract["categorical_features"]:
            if k not in vocabs:
                raise ValueError(f"vocabs_override missing {k!r}")
        validate_vocab_sizes_z(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(splits["train"], list(cfg["categorical_features"].keys()))
        validate_vocab_sizes_z(vocabs, contract)

    num_order = list(contract["x_numeric_zscore_column_order"])
    pcols = list(contract["player_constant_features"])
    wcol = cfg["weight_column"]

    train_df = splits["train"]
    p_stats_added: list[str] = []
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
            p_stats_added.append(c)

    circ_means, circ_stds = _build_circ_standardization(train_df, stats)

    train_a = dataframe_to_arrays_z_circular(
        splits["train"], stats, vocabs, num_order, wcol, circ_means, circ_stds
    )
    cal_a = dataframe_to_arrays_z_circular(
        splits["calibration"], stats, vocabs, num_order, wcol, circ_means, circ_stds
    )
    test_a = dataframe_to_arrays_z_circular(
        splits["test"], stats, vocabs, num_order, wcol, circ_means, circ_stds
    )

    meta = {
        "n_rows_train": len(splits["train"]),
        "n_rows_cal": len(splits["calibration"]),
        "n_rows_test": len(splits["test"]),
        "n_events_train": len(ev_t),
        "n_events_cal": len(ev_c),
        "n_events_test": len(ev_s),
        "vocabs": {k: len(v) for k, v in vocabs.items()},
        "player_constant_stats_from_train_only": p_stats_added,
        "target_stats_filled_train_only": stats_targets_added,
        "target_parameterization": "circular_6d",
        "circ_target_manifest": circular_target_transform_manifest(circ_means, circ_stds),
    }
    return train_a, cal_a, test_a, {"vocabs": vocabs, "meta": meta, "contract": contract}


def native_circular_target_manifest(gauss_means: np.ndarray, gauss_stds: np.ndarray) -> dict[str, Any]:
    return {
        "target_parameterization": "native_circular_vm",
        "angular_units_internal": "radians_wrapped_to_pi",
        "gaussian_block_columns": ["x", "e_y_star"],
        "gauss_means_train_stats": gauss_means.astype(float).tolist(),
        "gauss_stds_train_stats": gauss_stds.astype(float).tolist(),
        "angular_columns_degrees_on_disk": ["psi_deg", "theta_deg"],
        "note": "psi/theta trained as von Mises on wrapped radians; not sin/cos Gaussian.",
    }


def dataframe_to_arrays_z_native_circular(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    num_order: list[str],
    weight_col: str,
    gauss_means: np.ndarray,
    gauss_stds: np.ndarray,
) -> ZArraysNativeCircular:
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

    targets = list(Z_TARGET_COLUMNS)
    y_raw_four = np.stack(
        [pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64) for t in targets],
        axis=1,
    )
    x_raw = y_raw_four[:, 0]
    ey_raw = y_raw_four[:, 2]
    g_mu = gauss_means
    g_sig = np.maximum(gauss_stds, 1e-8)
    y_gauss = np.stack([(x_raw - g_mu[0]) / g_sig[0], (ey_raw - g_mu[1]) / g_sig[1]], axis=1).astype(np.float32)

    psi_deg = y_raw_four[:, 1]
    theta_deg = y_raw_four[:, 3]
    psi_rad = _wrap_pi(np.radians(psi_deg.astype(np.float64)))
    theta_rad = _wrap_pi(np.radians(theta_deg.astype(np.float64)))

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev = df["event_id"].to_numpy(dtype=np.int64)

    if np.isnan(x_num).any() or np.isnan(y_gauss).any():
        raise ValueError("NaN after native-circular z arrays")

    return ZArraysNativeCircular(
        x_num=x_num,
        cat=cat,
        y_gauss=y_gauss.astype(np.float32),
        psi_rad=psi_rad.astype(np.float32),
        theta_rad=theta_rad.astype(np.float32),
        y_raw_four=y_raw_four.astype(np.float32),
        w=w.astype(np.float32),
        event_id=ev,
        gauss_means=gauss_means.astype(np.float64),
        gauss_stds=gauss_stds.astype(np.float64),
        feature_names=list(num_order),
    )


def dataframe_to_arrays_z_native_trunc_ex(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    num_order: list[str],
    weight_col: str,
    gauss_means: np.ndarray,
    gauss_stds: np.ndarray,
    *,
    target_cols: tuple[str, str, str, str] = ("x", "psi_deg", "e_y_star", "e_x"),
) -> ZArraysNativeTruncEx:
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

    x_raw = pd.to_numeric(df[target_cols[0]], errors="coerce").to_numpy(dtype=np.float64)
    psi_deg = pd.to_numeric(df[target_cols[1]], errors="coerce").to_numpy(dtype=np.float64)
    ey_raw = pd.to_numeric(df[target_cols[2]], errors="coerce").to_numpy(dtype=np.float64)
    ex_raw = pd.to_numeric(df[target_cols[3]], errors="coerce").to_numpy(dtype=np.float64)

    g_mu = gauss_means
    g_sig = np.maximum(gauss_stds, 1e-8)
    y_gauss = np.stack([(x_raw - g_mu[0]) / g_sig[0], (ey_raw - g_mu[1]) / g_sig[1]], axis=1).astype(np.float32)
    psi_rad = _wrap_pi(np.radians(psi_deg))

    y_raw_targets = np.column_stack([x_raw, psi_deg, ey_raw, ex_raw]).astype(np.float32)

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev = df["event_id"].to_numpy(dtype=np.int64)

    if np.isnan(x_num).any() or np.isnan(y_gauss).any() or np.isnan(ex_raw).any():
        raise ValueError("NaN after native trunc-ex z arrays")

    return ZArraysNativeTruncEx(
        x_num=x_num,
        cat=cat,
        y_gauss=y_gauss.astype(np.float32),
        psi_rad=psi_rad.astype(np.float32),
        e_x=ex_raw.astype(np.float32),
        y_raw_targets=y_raw_targets.astype(np.float32),
        w=w.astype(np.float32),
        event_id=ev,
        gauss_means=gauss_means.astype(np.float64),
        gauss_stds=gauss_stds.astype(np.float64),
        feature_names=list(num_order),
    )


def load_all_z_data_native_circular(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
    feature_contract: dict[str, Any] | None = None,
) -> tuple[ZArraysNativeCircular, ZArraysNativeCircular, ZArraysNativeCircular, dict[str, Any]]:
    """Same splits/inputs as other z loaders; Gaussian block standardized; angles in radians.

    If ``feature_contract`` is set (e.g. a run's ``feature_contract_frozen.json``), it is used
    instead of resolving ``cfg["paths"]["feature_contract"]`` on disk — so inference matches the
    frozen semantics of that checkpoint even if the repo contract file later changes.
    """
    root = _project_root(project_root)
    contract = feature_contract if feature_contract is not None else load_feature_contract_z(cfg, root)
    validate_stage_z_config_against_contract(cfg, contract)

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

    split_table_path = root / paths.get("event_split_table", "data_processed/event_split_table.parquet")
    if split_table_path.is_file():
        spl = pd.read_parquet(split_table_path)
        for name, evset, key in [
            ("train", ev_t, "train"),
            ("calibration", ev_c, "calibration"),
            ("test", ev_s, "test"),
        ]:
            allowed = set(spl.loc[spl["split"].eq(key), "event_id"].unique())
            if not evset.issubset(allowed):
                bad = len(evset - allowed)
                if bad > 0:
                    raise ValueError(f"{name}: {bad} events not in official split table")

    stats_targets_added: list[str] = []
    train_df_for_tgt = splits["train"]
    for t in contract["targets"]:
        if t not in stats:
            v = pd.to_numeric(train_df_for_tgt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
            stats_targets_added.append(str(t))

    validate_preprocessing_stats_z(stats, contract)

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        for k in contract["categorical_features"]:
            if k not in vocabs:
                raise ValueError(f"vocabs_override missing {k!r}")
        validate_vocab_sizes_z(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(splits["train"], list(cfg["categorical_features"].keys()))
        validate_vocab_sizes_z(vocabs, contract)

    num_order = list(contract["x_numeric_zscore_column_order"])
    pcols = list(contract["player_constant_features"])
    wcol = cfg["weight_column"]

    train_df = splits["train"]
    p_stats_added: list[str] = []
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
            p_stats_added.append(c)

    for key in ("x", "e_y_star"):
        if key not in stats:
            raise ValueError(f"native circular requires {key!r} in standardization stats")
    gauss_means = np.array([stats["x"]["mean"], stats["e_y_star"]["mean"]], dtype=np.float64)
    gauss_stds = np.array(
        [max(float(stats["x"]["std"]), 1e-8), max(float(stats["e_y_star"]["std"]), 1e-8)],
        dtype=np.float64,
    )

    train_a = dataframe_to_arrays_z_native_circular(
        splits["train"], stats, vocabs, num_order, wcol, gauss_means, gauss_stds
    )
    cal_a = dataframe_to_arrays_z_native_circular(
        splits["calibration"], stats, vocabs, num_order, wcol, gauss_means, gauss_stds
    )
    test_a = dataframe_to_arrays_z_native_circular(
        splits["test"], stats, vocabs, num_order, wcol, gauss_means, gauss_stds
    )

    meta = {
        "n_rows_train": len(splits["train"]),
        "n_rows_cal": len(splits["calibration"]),
        "n_rows_test": len(splits["test"]),
        "n_events_train": len(ev_t),
        "n_events_cal": len(ev_c),
        "n_events_test": len(ev_s),
        "vocabs": {k: len(v) for k, v in vocabs.items()},
        "player_constant_stats_from_train_only": p_stats_added,
        "target_stats_filled_train_only": stats_targets_added,
        "target_parameterization": "native_circular_vm",
        "native_target_manifest": native_circular_target_manifest(gauss_means, gauss_stds),
    }
    return train_a, cal_a, test_a, {"vocabs": vocabs, "meta": meta, "contract": contract}


def load_all_z_data_native_trunc_ex(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
    feature_contract: dict[str, Any] | None = None,
) -> tuple[ZArraysNativeTruncEx, ZArraysNativeTruncEx, ZArraysNativeTruncEx, dict[str, Any]]:
    """Load z splits for ``native_trunc_ex_vm``: targets (x, psi_deg, e_y_star, e_x) on disk."""
    root = _project_root(project_root)
    contract = feature_contract if feature_contract is not None else load_feature_contract_z(cfg, root)
    validate_stage_z_config_against_contract(cfg, contract)
    targets = tuple(contract["targets"])
    if targets != ("x", "psi_deg", "e_y_star", "e_x"):
        raise ValueError(
            f"native_trunc_ex_vm expects targets (x, psi_deg, e_y_star, e_x); got {targets}. "
            "Rebuild z_train with e_x from posterior exports (see dataset_builders chunk_z)."
        )

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

    split_table_path = root / paths.get("event_split_table", "data_processed/event_split_table.parquet")
    if split_table_path.is_file():
        spl = pd.read_parquet(split_table_path)
        for name, evset, key in [
            ("train", ev_t, "train"),
            ("calibration", ev_c, "calibration"),
            ("test", ev_s, "test"),
        ]:
            allowed = set(spl.loc[spl["split"].eq(key), "event_id"].unique())
            if not evset.issubset(allowed):
                bad = len(evset - allowed)
                if bad > 0:
                    raise ValueError(f"{name}: {bad} events not in official split table")

    stats_targets_added: list[str] = []
    train_df_for_tgt = splits["train"]
    for t in contract["targets"]:
        if t not in stats:
            v = pd.to_numeric(train_df_for_tgt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
            stats_targets_added.append(str(t))

    validate_preprocessing_stats_z(stats, contract)

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        for k in contract["categorical_features"]:
            if k not in vocabs:
                raise ValueError(f"vocabs_override missing {k!r}")
        validate_vocab_sizes_z(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(splits["train"], list(cfg["categorical_features"].keys()))
        validate_vocab_sizes_z(vocabs, contract)

    num_order = list(contract["x_numeric_zscore_column_order"])
    pcols = list(contract["player_constant_features"])
    wcol = cfg["weight_column"]

    train_df = splits["train"]
    p_stats_added: list[str] = []
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
            p_stats_added.append(c)

    for key in ("x", "e_y_star"):
        if key not in stats:
            raise ValueError(f"native trunc-ex requires {key!r} in standardization stats")
    gauss_means = np.array([stats["x"]["mean"], stats["e_y_star"]["mean"]], dtype=np.float64)
    gauss_stds = np.array(
        [max(float(stats["x"]["std"]), 1e-8), max(float(stats["e_y_star"]["std"]), 1e-8)],
        dtype=np.float64,
    )

    train_a = dataframe_to_arrays_z_native_trunc_ex(
        splits["train"], stats, vocabs, num_order, wcol, gauss_means, gauss_stds, target_cols=targets
    )
    cal_a = dataframe_to_arrays_z_native_trunc_ex(
        splits["calibration"], stats, vocabs, num_order, wcol, gauss_means, gauss_stds, target_cols=targets
    )
    test_a = dataframe_to_arrays_z_native_trunc_ex(
        splits["test"], stats, vocabs, num_order, wcol, gauss_means, gauss_stds, target_cols=targets
    )

    meta = {
        "n_rows_train": len(splits["train"]),
        "n_rows_cal": len(splits["calibration"]),
        "n_rows_test": len(splits["test"]),
        "n_events_train": len(ev_t),
        "n_events_cal": len(ev_c),
        "n_events_test": len(ev_s),
        "vocabs": {k: len(v) for k, v in vocabs.items()},
        "player_constant_stats_from_train_only": p_stats_added,
        "target_stats_filled_train_only": stats_targets_added,
        "target_parameterization": "native_trunc_ex_vm",
        "native_target_manifest": native_circular_target_manifest(gauss_means, gauss_stds),
        "z_targets": list(targets),
    }
    return train_a, cal_a, test_a, {"vocabs": vocabs, "meta": meta, "contract": contract}
