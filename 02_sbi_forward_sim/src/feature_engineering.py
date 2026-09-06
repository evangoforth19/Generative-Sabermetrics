"""Engineered features and train-only standardization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    G_COLUMNS,
    P_COLUMNS,
    U_COLUMNS,
    Z_TARGET_COLUMNS,
)

STATCAST_RELEASE_SPEED_KEYS = ("game_date", "batter", "pitcher", "pitch_number")


def add_spin_axis_trig(df: pd.DataFrame, axis_col: str = "spin_axis") -> pd.DataFrame:
    out = df.copy()
    if axis_col not in out.columns:
        out["spin_axis_sin"] = np.nan
        out["spin_axis_cos"] = np.nan
        return out
    rad = np.deg2rad(pd.to_numeric(out[axis_col], errors="coerce"))
    out["spin_axis_sin"] = np.sin(rad)
    out["spin_axis_cos"] = np.cos(rad)
    return out


def merge_statcast_release_speed(df: pd.DataFrame, pickle_path: Path, *, table_name: str = "context") -> pd.DataFrame:
    """
    Attach true Statcast ``release_speed`` (mph) keyed by (game_date, batter, pitcher, pitch_number).

    Duplicate Statcast rows are deduplicated by preferring rows with a numeric release_speed (same keys
    can appear more than once in raw extracts).
    """
    out = df.copy()
    missing = [k for k in STATCAST_RELEASE_SPEED_KEYS if k not in out.columns]
    if missing:
        raise ValueError(
            f"{table_name}: cannot merge Statcast release_speed; missing merge key columns {missing}"
        )
    if not pickle_path.is_file():
        raise FileNotFoundError(
            f"{table_name}: Statcast pickle not found at {pickle_path.resolve()}. "
            "Provide the batter_data pickle containing column release_speed."
        )
    if "release_speed" in out.columns:
        raise ValueError(
            f"{table_name}: unexpected column release_speed before Statcast merge (ambiguous provenance)"
        )

    sc = pd.read_pickle(pickle_path)
    need = list(STATCAST_RELEASE_SPEED_KEYS) + ["release_speed"]
    miss_sc = [c for c in need if c not in sc.columns]
    if miss_sc:
        raise ValueError(f"Statcast pickle {pickle_path} missing columns {miss_sc}")

    sc = sc[list(STATCAST_RELEASE_SPEED_KEYS) + ["release_speed"]].copy()
    for k in STATCAST_RELEASE_SPEED_KEYS:
        sc[k] = pd.to_numeric(sc[k], errors="coerce").astype("Int64")
        out[k] = pd.to_numeric(out[k], errors="coerce").astype("Int64")
    sc["_rs_ok"] = pd.to_numeric(sc["release_speed"], errors="coerce").notna()
    sc = sc.sort_values("_rs_ok", ascending=False).drop_duplicates(
        subset=list(STATCAST_RELEASE_SPEED_KEYS), keep="first"
    ).drop(columns=["_rs_ok"])

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")

    merged = out.merge(sc, on=list(STATCAST_RELEASE_SPEED_KEYS), how="left")
    bad = merged["release_speed"].isna() | ~np.isfinite(pd.to_numeric(merged["release_speed"], errors="coerce"))
    if bad.any():
        ex = merged.loc[bad, ["event_id"] + list(STATCAST_RELEASE_SPEED_KEYS)].head(15)
        raise ValueError(
            f"{table_name}: {int(bad.sum())} rows lack Statcast release_speed after merge. "
            f"Example rows:\n{ex.to_string(index=False)}"
        )
    return merged


def attach_release_speed_from_kinematics(df: pd.DataFrame, *, table_name: str = "context") -> pd.DataFrame:
    """
    When no Statcast pickle is available, set ``release_speed`` (mph) from release velocity magnitude.

    Uses Statcast convention: ``vx0``, ``vy0``, ``vz0`` in ft/s at release; mph = ||v|| * (3600/5280).
    """
    out = df.copy()
    if "release_speed" in out.columns:
        raise ValueError(f"{table_name}: unexpected column release_speed before kinematics attach")
    need = ("vx0", "vy0", "vz0")
    miss = [c for c in need if c not in out.columns]
    if miss:
        raise ValueError(
            f"{table_name}: cannot derive release_speed without Statcast pickle; missing columns {miss}"
        )
    v = np.stack(
        [
            pd.to_numeric(out["vx0"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(out["vy0"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(out["vz0"], errors="coerce").to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    mag_fps = np.linalg.norm(v, axis=1)
    out["release_speed"] = mag_fps * (3600.0 / 5280.0)
    bad = ~np.isfinite(out["release_speed"].to_numpy())
    if bad.any():
        n = int(bad.sum())
        ex = out.loc[bad, ["event_id"] + list(need)].head(15)
        raise ValueError(
            f"{table_name}: {n} rows have non-finite derived release_speed. Examples:\n{ex.to_string(index=False)}"
        )
    return out


def z_count_from_balls_strikes(df: pd.DataFrame, *, table_name: str = "context") -> pd.DataFrame:
    """Require z_count from production context, or reconstruct from balls/strikes via the official map (no partial NA)."""
    out = df.copy()
    if "z_count" in out.columns:
        if not out["z_count"].notna().all():
            n = int(out["z_count"].isna().sum())
            raise ValueError(f"{table_name}: z_count has {n} missing values; refusing to guess.")
        return out

    z_count_map = {
        "0-0": 0.3144746543,
        "0-1": 0.2704261285,
        "0-2": 0.2004055518,
        "1-0": 0.3612777549,
        "1-1": 0.3050139272,
        "1-2": 0.2260470281,
        "2-0": 0.4309212696,
        "2-1": 0.3613989554,
        "2-2": 0.2734161604,
        "3-0": 0.5416601816,
        "3-1": 0.4803511791,
        "3-2": 0.3824951644,
    }
    if "balls" not in out.columns or "strikes" not in out.columns:
        raise ValueError(f"{table_name}: z_count absent and balls/strikes unavailable.")
    cs = (
        out["balls"].astype("Int64").astype(str) + "-" + out["strikes"].astype("Int64").astype(str)
    )
    out["z_count"] = cs.map(z_count_map)
    if out["z_count"].isna().any():
        bad = sorted(cs[out["z_count"].isna()].dropna().unique().tolist())
        raise ValueError(f"{table_name}: unmapped count_str values for z_count: {bad}")
    return out


def compute_train_standardization(
    df_train: pd.DataFrame,
    numeric_cols: list[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for c in numeric_cols:
        if c not in df_train.columns:
            continue
        s = pd.to_numeric(df_train[c], errors="coerce")
        mu = float(s.mean(skipna=True))
        sig = float(s.std(skipna=True))
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        stats[c] = {"mean": mu, "std": sig}
    return stats


def apply_standardization(df: pd.DataFrame, stats: dict[str, dict[str, float]], suffix: str = "_z") -> pd.DataFrame:
    out = df.copy()
    for c, d in stats.items():
        if c not in out.columns:
            continue
        col = f"{c}{suffix}"
        out[col] = (pd.to_numeric(out[c], errors="coerce") - d["mean"]) / d["std"]
    return out


def save_standardization_stats(stats: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_standardization_stats(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def standardization_numeric_blocks(
    df: pd.DataFrame,
) -> list[str]:
    """Columns to standardize: numeric g, p, u, z stage targets (not y for simulator direction)."""
    cols = []
    for block in (G_COLUMNS, P_COLUMNS, U_COLUMNS):
        for c in block:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    for c in Z_TARGET_COLUMNS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if "e_x" in df.columns and pd.api.types.is_numeric_dtype(df["e_x"]):
        cols.append("e_x")
    return sorted(set(cols))

