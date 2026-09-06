"""Event-level splits without draw leakage; approximate stratification by batter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def event_stratified_split(
    event_ids: np.ndarray,
    batter_names: np.ndarray,
    train_frac: float = 0.70,
    cal_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split events so all draws for an event share one split.

    For each batter, allocate that batter's events 70/15/15 (with floor rounding);
    remainder events assigned to train to avoid empty splits for small batters.
    """
    if not np.isclose(train_frac + cal_frac + test_frac, 1.0):
        raise ValueError("train + cal + test must sum to 1")

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"event_id": event_ids, "batter_name": batter_names}).drop_duplicates(
        "event_id"
    )

    assignments: dict[int, str] = {}
    for _, g in frame.groupby("batter_name"):
        eids = g["event_id"].to_numpy()
        rng.shuffle(eids)
        n = len(eids)
        n_train = int(np.floor(train_frac * n))
        n_cal = int(np.floor(cal_frac * n))
        n_test = n - n_train - n_cal
        if n_train <= 0:
            n_train = min(1, n)
            n_cal = min(n_cal, n - n_train)
            n_test = n - n_train - n_cal
        splits = ["train"] * n_train + ["calibration"] * n_cal + ["test"] * n_test
        if len(splits) < n:
            splits.extend(["train"] * (n - len(splits)))
        splits = splits[:n]
        for eid, sp in zip(eids, splits):
            assignments[int(eid)] = sp

    out = pd.DataFrame({"event_id": frame["event_id"], "batter_name": frame["batter_name"]})
    out["split"] = out["event_id"].map(assignments)
    return out


def save_split_manifest(
    split_table: pd.DataFrame,
    path: Path,
    train_frac: float,
    cal_frac: float,
    test_frac: float,
    seed: int,
) -> None:
    counts = split_table["split"].value_counts().to_dict()
    payload = {
        "train_frac": train_frac,
        "calibration_frac": cal_frac,
        "test_frac": test_frac,
        "seed": seed,
        "n_events": int(split_table["event_id"].nunique()),
        "counts_by_split": {k: int(v) for k, v in counts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def attach_split_to_draws(df: pd.DataFrame, split_table: pd.DataFrame) -> pd.DataFrame:
    m = split_table.set_index("event_id")["split"]
    out = df.copy()
    out["split"] = out["event_id"].map(m)
    missing = out["split"].isna().sum()
    if missing:
        raise ValueError(f"{missing} draws missing split assignment")
    return out


def attach_split_to_events(df: pd.DataFrame, split_table: pd.DataFrame, id_col: str = "event_id") -> pd.DataFrame:
    return df.merge(split_table[[id_col, "split"]], on=id_col, how="left")
