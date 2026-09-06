"""Spray-angle (d_tilde / SA) support used in forward simulation and bootstrap empirical pools."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPRAY_ANGLE_DEG_LIMITS: tuple[float, float] = (-45.0, 45.0)


def spray_angle_in_support(
    deg: float | np.ndarray,
    *,
    limits: tuple[float, float] = SPRAY_ANGLE_DEG_LIMITS,
) -> np.ndarray | bool:
    """True where signed spray angle lies in ``limits`` (inclusive)."""
    lo, hi = float(limits[0]), float(limits[1])
    d = np.asarray(deg, dtype=np.float64)
    ok = (d >= lo) & (d <= hi) & np.isfinite(d)
    if ok.ndim == 0:
        return bool(ok)
    return ok


def filter_spray_angle_deg(
    series: pd.Series,
    *,
    limits: tuple[float, float] = SPRAY_ANGLE_DEG_LIMITS,
) -> pd.Series:
    """Boolean mask: finite spray angle within ``limits``."""
    sa = pd.to_numeric(series, errors="coerce")
    lo, hi = limits
    return sa.notna() & (sa >= lo) & (sa <= hi)


def filter_bip_model_eligible(
    df: pd.DataFrame,
    *,
    limits: tuple[float, float] = SPRAY_ANGLE_DEG_LIMITS,
    spray_col: str | None = None,
) -> pd.DataFrame:
    """
    Rows eligible under ``bip_model`` QC + fair wedge and spray angle in ``limits``.

    Expects production ``context_event_table`` / ``sbi_context_event_master`` columns.
    """
    from .dataset_builders import bip_model_training_event_mask

    if spray_col is None:
        if "spray_angle_deg" in df.columns:
            spray_col = "spray_angle_deg"
        elif "spray_angle_obs_deg" in df.columns:
            spray_col = "spray_angle_obs_deg"
        else:
            raise ValueError("df missing spray_angle_deg / spray_angle_obs_deg")

    m = bip_model_training_event_mask(df) & filter_spray_angle_deg(df[spray_col], limits=limits)
    return df.loc[m].copy()
