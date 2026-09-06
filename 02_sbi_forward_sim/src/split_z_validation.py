"""Split z_calibration into val_select (model selection) and temp_cal (temperature fit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .split_u_validation import partition_df_by_events, split_calibration_events


def save_z_validation_split_metadata(
    out_dir: Path,
    val_event_ids: set[int],
    temp_event_ids: set[int],
    summary: dict[str, Any],
) -> None:
    meta_dir = out_dir / "split_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "z_val_select_event_ids.txt").write_text(
        "\n".join(str(e) for e in sorted(val_event_ids)),
        encoding="utf-8",
    )
    (meta_dir / "z_temp_cal_event_ids.txt").write_text(
        "\n".join(str(e) for e in sorted(temp_event_ids)),
        encoding="utf-8",
    )
    (meta_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def split_z_calibration_dataframe(
    cal_df: pd.DataFrame,
    *,
    val_fraction: float = 0.5,
    seed: int = 20260502,
    event_col: str = "event_id",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Event-level split of calibration rows into val_select and temp_cal."""
    val_ev, temp_ev, base = split_calibration_events(
        cal_df, val_fraction=val_fraction, seed=seed, event_col=event_col
    )
    val_df = partition_df_by_events(cal_df, val_ev, event_col)
    temp_df = partition_df_by_events(cal_df, temp_ev, event_col)
    summary = dict(base)
    summary["use_event_level_split"] = True
    return val_df, temp_df, summary


__all__ = [
    "save_z_validation_split_metadata",
    "split_z_calibration_dataframe",
    "split_calibration_events",
    "partition_df_by_events",
]
