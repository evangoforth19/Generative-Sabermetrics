"""Load, validate, and tensorize stage-u parquet datasets."""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_contract_u import (
    load_feature_contract,
    validate_preprocessing_stats_against_contract,
    validate_stage_u_config_against_contract,
    validate_vocab_sizes_against_contract,
)
from .player_target_bounds_u import (
    BOUNDS_FIT_WARNING,
    fit_player_target_bounds_train_only,
    bounds_diagnostics_global_and_player,
    save_bounds_artifacts,
)
from .split_u_validation import (
    partition_df_by_events,
    save_validation_split_metadata,
    split_calibration_events,
)
from .target_transforms_u import (
    BoundedLogitZScoreTransform,
    PlayerSpecificBoundedLogitZScoreTransform,
    bounds_diagnostics_for_split,
    uses_player_specific_bounded_transform,
    warn_if_train_oob_exceeds,
)

U_TARGETS = ("v_ss_tilde", "a_tilde", "d_tilde")
REQUIRED_INPUT_NUMERIC = (
    "z_count",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "spin_axis_sin",
    "spin_axis_cos",
    "release_speed",
)
P_CONSTANTS = (
    "bat_length_in",
    "bat_weight_oz",
    "x_cm_fixed_in",
    "r_g_fixed_in",
    "I0_oz_in2",
    "Iz_oz_in2",
)
CAT_COLS = ("batter_name", "pitch_type", "stand", "p_throws")
FORBIDDEN_U_COLS = ("phi_star", "theta_deg", "theta")


def _resolve_project_root(project_root: Path | None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def load_standardization_stats(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_u_dataframe(
    df: pd.DataFrame,
    split_name: str,
    *,
    require_weight: bool = True,
    weight_col: str = "combined_training_weight",
) -> None:
    for c in U_TARGETS:
        if c not in df.columns:
            raise ValueError(f"{split_name}: missing target {c!r}")
    for c in REQUIRED_INPUT_NUMERIC:
        if c not in df.columns:
            raise ValueError(f"{split_name}: missing input {c!r}")
    for c in FORBIDDEN_U_COLS:
        if c in df.columns:
            raise ValueError(f"{split_name}: forbidden column {c!r} present for stage-u")
    ambiguous = [c for c in ("theta_obs_deg", "theta_star_deg") if c in df.columns]
    if ambiguous:
        raise ValueError(
            f"{split_name}: θ fields {ambiguous} must not appear in stage-u table"
        )
    if require_weight and weight_col not in df.columns:
        raise ValueError(f"{split_name}: missing weight column {weight_col!r}")
    if "release_speed" in df.columns:
        if df["release_speed"].isna().mean() > 0.05:
            raise ValueError(f"{split_name}: release_speed has too many NA (>5%)")
    if "spin_axis_sin" not in df.columns or "spin_axis_cos" not in df.columns:
        raise ValueError(f"{split_name}: spin_axis sin/cos required")


def validate_no_event_leakage(
    ev_train: set[int],
    ev_cal: set[int],
    ev_test: set[int],
) -> None:
    if ev_train & ev_cal:
        raise ValueError(f"train/cal event leakage: {len(ev_train & ev_cal)} events")
    if ev_train & ev_test:
        raise ValueError(f"train/test event leakage: {len(ev_train & ev_test)} events")
    if ev_cal & ev_test:
        raise ValueError(f"cal/test event leakage: {len(ev_cal & ev_test)} events")


def merge_player_constants(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id"] + list(P_CONSTANTS)
    miss = [c for c in cols if c not in master.columns]
    if miss:
        raise ValueError(f"context master missing columns {miss}")
    sub = master[cols].drop_duplicates("event_id")
    out = df.merge(sub, on="event_id", how="left", validate="many_to_one")
    for c in P_CONSTANTS:
        if out[c].isna().any():
            n = int(out[c].isna().sum())
            raise ValueError(f"After merge, {n} rows missing player constant {c!r}")
    return out


def build_categorical_vocabs(train_df: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        vals = train_df[col].fillna("NA").astype(str).unique().tolist()
        vals = sorted(vals)
        tok2idx = {"<UNK>": 0}
        for i, v in enumerate(vals, start=1):
            tok2idx[v] = i
        vocabs[col] = tok2idx
    return vocabs


def encode_categories(df: pd.DataFrame, col: str, tok2idx: dict[str, int]) -> np.ndarray:
    s = df[col].fillna("NA").astype(str).str.strip()
    return np.array([tok2idx.get(v, 0) for v in s], dtype=np.int64)


@dataclass
class UArrays:
    x_num: np.ndarray  # (N, F) standardized
    cat: dict[str, np.ndarray]  # col -> (N,) int64
    y: np.ndarray  # (N, 3) model-space targets (z-score or logit-z for va)
    y_raw: np.ndarray  # (N, 3) original scale
    w: np.ndarray  # (N,)
    event_id: np.ndarray  # (N,)
    target_means: np.ndarray  # (3,) legacy z-score for d; va if no transform
    target_stds: np.ndarray  # (3,)
    feature_names: list[str]
    va_transform: BoundedLogitZScoreTransform | PlayerSpecificBoundedLogitZScoreTransform | None = None
    batter_id: np.ndarray | None = None  # (N,) hierarchical player index


def _parse_va_bounds(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    tt = cfg.get("target_transform", {})
    raw = tt.get("bounds", {})
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def uses_bounded_va_transform(cfg: dict[str, Any]) -> bool:
    tt = cfg.get("target_transform", {})
    return tt.get("va_transform") == "bounded_logit_zscore"


def dataframe_to_arrays(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    numeric_cols: list[str],
    p_cols: list[str],
    weight_col: str,
    *,
    va_transform: BoundedLogitZScoreTransform | PlayerSpecificBoundedLogitZScoreTransform | None = None,
) -> UArrays:
    num_all = numeric_cols + p_cols
    for c in num_all:
        if c not in stats:
            raise ValueError(f"No standardization stats for feature {c!r}")
    x_parts = []
    names: list[str] = []
    for c in num_all:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        mu, sig = stats[c]["mean"], stats[c]["std"]
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        x_parts.append((v - mu) / sig)
        names.append(c)
    x_num = np.stack(x_parts, axis=1).astype(np.float32)

    cat: dict[str, np.ndarray] = {}
    for col in vocabs:
        cat[col] = encode_categories(df, col, vocabs[col])

    y_raw = np.stack(
        [pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64) for t in U_TARGETS],
        axis=1,
    )
    t_means = np.array([stats[t]["mean"] for t in U_TARGETS], dtype=np.float64)
    t_stds = np.array([max(stats[t]["std"], 1e-8) for t in U_TARGETS], dtype=np.float64)

    batter_id = None
    if "batter_name" in vocabs:
        batter_id = encode_categories(df, "batter_name", vocabs["batter_name"])

    if va_transform is not None:
        if batter_id is None:
            raise ValueError("va_transform requires batter_id")
        if isinstance(va_transform, PlayerSpecificBoundedLogitZScoreTransform):
            y_va = va_transform.transform(y_raw[:, :2], batter_id)
        else:
            y_va = va_transform.transform(y_raw[:, :2])
        y_d = (y_raw[:, 2] - t_means[2]) / t_stds[2]
        y = np.column_stack([y_va, y_d]).astype(np.float32)
    else:
        y = ((y_raw - t_means) / t_stds).astype(np.float32)

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ev = df["event_id"].to_numpy(dtype=np.int64)

    if np.isnan(x_num).any():
        raise ValueError("NaN in standardized numeric features after fill")
    if np.isnan(y).any():
        raise ValueError("NaN in targets")

    return UArrays(
        x_num=x_num,
        cat=cat,
        y=y.astype(np.float32),
        y_raw=y_raw.astype(np.float32),
        w=w.astype(np.float32),
        event_id=ev,
        target_means=t_means,
        target_stds=t_stds,
        feature_names=names,
        va_transform=va_transform,
        batter_id=batter_id,
    )


def index_by_event(event_ids: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    for i, e in enumerate(event_ids.tolist()):
        out.setdefault(int(e), []).append(i)
    return {e: np.array(ix, dtype=np.int64) for e, ix in out.items()}


def sample_event_balanced_indices(
    idx_by_event: dict[int, np.ndarray],
    event_list: list[int],
    n_events: int,
    max_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_events > len(event_list):
        chosen = event_list
    else:
        chosen = rng.choice(event_list, size=n_events, replace=False).tolist()
    blocks: list[np.ndarray] = []
    for e in chosen:
        pool = idx_by_event[e]
        if len(pool) <= max_draws:
            blocks.append(pool)
        else:
            blocks.append(rng.choice(pool, size=max_draws, replace=False))
    return np.concatenate(blocks, axis=0)


def load_all_u_data(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
) -> tuple[UArrays, UArrays, UArrays, dict[str, Any]]:
    root = _resolve_project_root(project_root)
    paths = cfg["paths"]
    contract = load_feature_contract(cfg, root)
    validate_stage_u_config_against_contract(cfg, contract)

    stats: dict[str, dict[str, float]] = copy.deepcopy(
        load_standardization_stats(root / paths["standardization_stats"])
    )
    validate_preprocessing_stats_against_contract(stats, contract)
    master = pd.read_parquet(root / paths["context_master"])

    splits_dfs: dict[str, pd.DataFrame] = {}
    for split, key in [("train", "u_train"), ("calibration", "u_calibration"), ("test", "u_test")]:
        df = pd.read_parquet(root / paths[key])
        validate_u_dataframe(df, split, weight_col=cfg["weight_column"])
        df = merge_player_constants(df, master)
        splits_dfs[split] = df

    ev_t = set(splits_dfs["train"]["event_id"].unique())
    ev_c = set(splits_dfs["calibration"]["event_id"].unique())
    ev_s = set(splits_dfs["test"]["event_id"].unique())
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
                    raise ValueError(f"{name}: {bad} events not in official split table for {key}")

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        for k in contract["categorical_features"]:
            if k not in vocabs:
                raise ValueError(f"vocabs_override missing categorical column {k!r}")
        validate_vocab_sizes_against_contract(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(
            splits_dfs["train"],
            list(cfg["categorical_features"].keys()),
        )
        validate_vocab_sizes_against_contract(vocabs, contract)

    num_cols = list(cfg["numeric_features"])
    pcols = list(cfg["player_constant_features"])
    wcol = cfg["weight_column"]

    train_df = splits_dfs["train"]
    p_stats_added: list[str] = []
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
            p_stats_added.append(c)

    va_transform: BoundedLogitZScoreTransform | None = None
    bounds_diag: dict[str, Any] = {}
    if uses_bounded_va_transform(cfg):
        bounds = _parse_va_bounds(cfg)
        eps = float(cfg.get("target_transform", {}).get("eps", 1e-5))
        for split_name, df in splits_dfs.items():
            bounds_diag[split_name] = bounds_diagnostics_for_split(
                df, bounds, split_name, eps=eps
            )
        for wmsg in warn_if_train_oob_exceeds(bounds_diag["train"], threshold_pct=0.5):
            print(wmsg)
        train_raw_va = splits_dfs["train"][list(("v_ss_tilde", "a_tilde"))].to_numpy(dtype=np.float64)
        train_w = splits_dfs["train"][wcol].to_numpy(dtype=np.float64)
        va_transform = BoundedLogitZScoreTransform(bounds=bounds, eps=eps)
        va_transform.fit(train_raw_va, sample_weight=train_w)

    train_a = dataframe_to_arrays(
        splits_dfs["train"], stats, vocabs, num_cols, pcols, wcol, va_transform=va_transform
    )
    cal_a = dataframe_to_arrays(
        splits_dfs["calibration"], stats, vocabs, num_cols, pcols, wcol, va_transform=va_transform
    )
    test_a = dataframe_to_arrays(
        splits_dfs["test"], stats, vocabs, num_cols, pcols, wcol, va_transform=va_transform
    )

    meta = {
        "n_rows_train": len(splits_dfs["train"]),
        "n_rows_cal": len(splits_dfs["calibration"]),
        "n_rows_test": len(splits_dfs["test"]),
        "n_events_train": len(ev_t),
        "n_events_cal": len(ev_c),
        "n_events_test": len(ev_s),
        "vocabs": {k: len(v) for k, v in vocabs.items()},
        "player_constant_stats_from_train_only": p_stats_added,
        "bounds_diagnostics": bounds_diag,
        "target_transform_type": (
            "bounded_logit_zscore" if va_transform is not None else "zscore_all_targets"
        ),
    }
    return train_a, cal_a, test_a, {"vocabs": vocabs, "meta": meta, "va_transform": va_transform}


def load_all_u_data_player_support(
    cfg: dict[str, Any],
    project_root: Path | None = None,
    *,
    vocabs_override: dict[str, dict[str, int]] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load train / val_select / temp_cal / test with train-only player bounds.

    u_calibration is split by event into val_select (early stop) and temp_cal (temperatures).
    """
    root = _resolve_project_root(project_root)
    paths = cfg["paths"]
    contract = load_feature_contract(cfg, root)
    validate_stage_u_config_against_contract(cfg, contract)

    stats = copy.deepcopy(load_standardization_stats(root / paths["standardization_stats"]))
    validate_preprocessing_stats_against_contract(stats, contract)
    master = pd.read_parquet(root / paths["context_master"])

    train_df = pd.read_parquet(root / paths["u_train"])
    cal_df = pd.read_parquet(root / paths["u_calibration"])
    test_df = pd.read_parquet(root / paths["u_test"])
    wcol = cfg["weight_column"]
    for name, df in [("train", train_df), ("calibration", cal_df), ("test", test_df)]:
        validate_u_dataframe(df, name, weight_col=wcol)
    train_df = merge_player_constants(train_df, master)
    cal_df = merge_player_constants(cal_df, master)
    test_df = merge_player_constants(test_df, master)

    d_clip_diag: dict[str, Any] = {}
    d_cap_cfg = cfg.get("bounds", {}).get("player_specific", {})
    if bool(d_cap_cfg.get("clip_d_tilde_to_global_cap", False)):
        raw_cap = d_cap_cfg.get(
            "d_tilde_global_cap",
            d_cap_cfg.get("attack_direction_global_cap", [-45.0, 45.0]),
        )
        d_low, d_high = float(raw_cap[0]), float(raw_cap[1])

        def _clip_d(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
            out = df.copy()
            d = pd.to_numeric(out["d_tilde"], errors="coerce")
            below = d < d_low
            above = d > d_high
            clipped = below | above
            d_clip_diag[split_name] = {
                "bounds": [d_low, d_high],
                "n_rows": int(len(out)),
                "below": int(below.sum()),
                "above": int(above.sum()),
                "pct_clipped": 100.0 * int(clipped.sum()) / max(int(len(out)), 1),
                "by_player": [
                    {
                        "player": str(player),
                        "n_rows": int(len(gdf)),
                        "pct_clipped": 100.0
                        * int(((pd.to_numeric(gdf["d_tilde"], errors="coerce") < d_low)
                               | (pd.to_numeric(gdf["d_tilde"], errors="coerce") > d_high)).sum())
                        / max(int(len(gdf)), 1),
                    }
                    for player, gdf in out.groupby("batter_name", sort=False)
                ],
            }
            out["d_tilde"] = d.clip(lower=d_low, upper=d_high)
            return out

        train_df = _clip_d(train_df, "train")
        cal_df = _clip_d(cal_df, "calibration")
        test_df = _clip_d(test_df, "test")

    vcfg = cfg.get("validation", {})
    val_frac = float(vcfg.get("val_select_fraction", 0.5))
    split_seed = int(vcfg.get("split_seed", 20260408))
    val_ev, temp_ev, split_summary = split_calibration_events(
        cal_df, val_fraction=val_frac, seed=split_seed
    )
    val_df = partition_df_by_events(cal_df, val_ev)
    temp_df = partition_df_by_events(cal_df, temp_ev)

    if vocabs_override is not None:
        vocabs = {k: dict(vocabs_override[k]) for k in vocabs_override}
        validate_vocab_sizes_against_contract(vocabs, contract)
    else:
        vocabs = build_categorical_vocabs(train_df, list(cfg["categorical_features"].keys()))
        validate_vocab_sizes_against_contract(vocabs, contract)

    print(BOUNDS_FIT_WARNING)
    player_bounds = fit_player_target_bounds_train_only(train_df, cfg, vocabs=vocabs)

    eps = float(cfg.get("target_transform", {}).get("eps", 1e-5))
    va_transform = PlayerSpecificBoundedLogitZScoreTransform(player_bounds, eps=eps)
    train_bid = encode_categories(train_df, "batter_name", vocabs["batter_name"])
    va_transform.fit(train_df, train_bid, sample_weight_col=wcol)

    num_cols = list(cfg["numeric_features"])
    pcols = list(cfg["player_constant_features"])
    for c in pcols:
        if c not in stats:
            v = pd.to_numeric(train_df[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}

    def _to_arrays(df: pd.DataFrame) -> UArrays:
        return dataframe_to_arrays(df, stats, vocabs, num_cols, pcols, wcol, va_transform=va_transform)

    train_a = _to_arrays(train_df)
    val_a = _to_arrays(val_df)
    temp_a = _to_arrays(temp_df)
    test_a = _to_arrays(test_df)

    diag_all: dict[str, Any] = {"global": {}, "player_rows": []}
    for split_name, df in [
        ("train", train_df),
        ("val_select", val_df),
        ("temp_cal", temp_df),
        ("test", test_df),
    ]:
        g, rows = bounds_diagnostics_global_and_player(
            df, player_bounds, split_name, vocabs=vocabs
        )
        diag_all["global"][split_name] = g
        diag_all["player_rows"].extend(rows)

    thresh = float(cfg.get("bounds", {}).get("train_clipping_warning_threshold", 0.01))
    allow = bool(cfg.get("bounds", {}).get("allow_train_clipping_above_threshold", True))
    for name in ("v_ss_tilde", "a_tilde"):
        pct = diag_all["global"]["train"]["targets"][name].get("pct_outside_global", 0)
        if pct > 100 * thresh:
            msg = f"WARNING: train {name} {pct:.2f}% outside fitted global bounds"
            print(msg)
            if not allow:
                raise ValueError(msg)

    if run_dir is not None:
        save_validation_split_metadata(run_dir, val_ev, temp_ev, split_summary)
        save_bounds_artifacts(player_bounds, diag_all, run_dir)
        if d_clip_diag:
            (run_dir / "d_tilde_clip_diagnostics.json").write_text(
                json.dumps(d_clip_diag, indent=2), encoding="utf-8"
            )

    meta = {
        "n_rows_train": len(train_df),
        "n_rows_val_select": len(val_df),
        "n_rows_temp_cal": len(temp_df),
        "n_rows_test": len(test_df),
        "n_events_train": int(train_df["event_id"].nunique()),
        "n_events_val_select": len(val_ev),
        "n_events_temp_cal": len(temp_ev),
        "n_events_test": int(test_df["event_id"].nunique()),
        "target_transform_type": "player_specific_bounded_logit_zscore",
        "bounds_fit_warning": BOUNDS_FIT_WARNING,
        "split_summary": split_summary,
        "d_tilde_clip_diagnostics": d_clip_diag,
    }
    return {
        "train": train_a,
        "val_select": val_a,
        "temp_cal": temp_a,
        "test": test_a,
        "vocabs": vocabs,
        "meta": meta,
        "va_transform": va_transform,
        "player_bounds": player_bounds,
        "bounds_diagnostics": diag_all,
    }
