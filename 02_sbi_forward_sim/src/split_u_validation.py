"""Split u_calibration into val_select (model selection) and temp_cal (temperature fit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def split_calibration_events(
    cal_df: pd.DataFrame,
    *,
    val_fraction: float = 0.5,
    seed: int = 20260408,
    event_col: str = "event_id",
) -> tuple[set[int], set[int], dict[str, Any]]:
    """
    Deterministic event-level split of calibration events.

    Returns (val_select_event_ids, temp_cal_event_ids, summary_dict).
    """
    events = np.array(sorted(cal_df[event_col].unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(events)
    n_val = max(1, int(np.floor(val_fraction * len(events))))
    n_val = min(n_val, len(events) - 1) if len(events) > 1 else len(events)
    val_ev = set(int(e) for e in events[:n_val])
    temp_ev = set(int(e) for e in events[n_val:])
    if not temp_ev and len(events) > 1:
        temp_ev = {int(events[-1])}
        val_ev.discard(int(events[-1]))
    summary = {
        "n_calibration_events_total": int(len(events)),
        "n_val_select_events": len(val_ev),
        "n_temp_cal_events": len(temp_ev),
        "val_select_fraction": val_fraction,
        "split_seed": seed,
        "n_val_select_rows": int(cal_df[cal_df[event_col].isin(val_ev)].shape[0]),
        "n_temp_cal_rows": int(cal_df[cal_df[event_col].isin(temp_ev)].shape[0]),
    }
    return val_ev, temp_ev, summary


def partition_df_by_events(df: pd.DataFrame, event_ids: set[int], event_col: str = "event_id") -> pd.DataFrame:
    return df.loc[df[event_col].isin(event_ids)].copy()


def save_validation_split_metadata(
    out_dir: Path,
    val_event_ids: set[int],
    temp_event_ids: set[int],
    summary: dict[str, Any],
) -> None:
    meta_dir = out_dir / "split_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "u_val_select_event_ids.txt").write_text(
        "\n".join(str(e) for e in sorted(val_event_ids)), encoding="utf-8"
    )
    (meta_dir / "u_temp_cal_event_ids.txt").write_text(
        "\n".join(str(e) for e in sorted(temp_event_ids)), encoding="utf-8"
    )
    (meta_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
