"""Bounded logit-zscore transforms for stage-z branch autoreg MDN (x, e_y_star, e_x)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def logit(s: np.ndarray, eps: float) -> np.ndarray:
    s = np.clip(s, eps, 1.0 - eps)
    return _safe_log(s) - _safe_log(1.0 - s)


def sigmoid(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(r, -50.0, 50.0)))


@dataclass
class BoundedLogitZscoreState:
    """Train-fitted global z-score stats on logit coordinates."""

    mean_r_x: float
    std_r_x: float
    mean_r_y: float
    std_r_y: float
    mean_r_ex: float
    std_r_ex: float
    eps: float
    x_support_mode: str
    x_used_empirical_fallback: bool
    e_y_lower: float
    e_y_upper: float
    e_x_lower: float
    e_x_upper: float
    e_x_scale: float  # upper raw bound (0.6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zscore_scope": "global_train_after_logit",
            "mean_r_x": self.mean_r_x,
            "std_r_x": self.std_r_x,
            "mean_r_y": self.mean_r_y,
            "std_r_y": self.std_r_y,
            "mean_r_ex": self.mean_r_ex,
            "std_r_ex": self.std_r_ex,
            "eps": self.eps,
            "x_support_mode": self.x_support_mode,
            "x_used_empirical_fallback": self.x_used_empirical_fallback,
            "e_y_star_bounds": [self.e_y_lower, self.e_y_upper],
            "e_x_bounds": [self.e_x_lower, self.e_x_upper],
            "e_x_scale": self.e_x_scale,
        }


def compute_x_bounds_per_row(
    bat_length_in: np.ndarray,
    *,
    cfg_x: dict[str, Any],
    train_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Returns (lower, upper, used_empirical_fallback, mode).
    """
    mode = str(cfg_x.get("support_mode", "physics_from_player_constants"))
    eps = float(cfg_x.get("eps", 1e-5))
    padding = float(cfg_x.get("fallback_padding_abs", 0.25))
    L = np.asarray(bat_length_in, dtype=np.float64)
    used_empirical = False
    if mode == "physics_from_player_constants":
        lower = L - 11.0
        upper = L
        span = upper - lower
        if np.all(span > 0):
            return lower.astype(np.float64), upper.astype(np.float64), False, "physics_from_player_constants"
        used_empirical = True
        mode = str(cfg_x.get("fallback_mode", "train_empirical_with_padding"))

    if train_x is None:
        raise ValueError("train_x required for empirical x bounds")
    lo_g = float(np.nanmin(train_x)) - padding
    hi_g = float(np.nanmax(train_x)) + padding
    lower = np.full_like(L, lo_g, dtype=np.float64)
    upper = np.full_like(L, hi_g, dtype=np.float64)
    return lower, upper, used_empirical, mode


def raw_to_model_coords(
    x: np.ndarray,
    e_y: np.ndarray,
    e_x: np.ndarray,
    *,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    state: BoundedLogitZscoreState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return r_x_std, r_y_std, r_ex_std and intermediate s for Jacobian."""
    eps = state.eps
    span_x = np.maximum(x_upper - x_lower, eps)
    s_x = (x - x_lower) / span_x
    s_x_clip = np.clip(s_x, eps, 1.0 - eps)
    r_x = logit(s_x_clip, eps)
    r_x_std = (r_x - state.mean_r_x) / state.std_r_x

    s_y = np.clip(e_y, state.e_y_lower, state.e_y_upper)
    s_y_clip = np.clip(s_y, eps, 1.0 - eps)
    r_y = logit(s_y_clip, eps)
    r_y_std = (r_y - state.mean_r_y) / state.std_r_y

    s_ex = e_x / state.e_x_scale
    s_ex_clip = np.clip(s_ex, eps, 1.0 - eps)
    r_ex = logit(s_ex_clip, eps)
    r_ex_std = (r_ex - state.mean_r_ex) / state.std_r_ex

    aux = {"s_x_clip": s_x_clip, "s_y_clip": s_y_clip, "s_ex_clip": s_ex_clip, "span_x": span_x}
    return (
        r_x_std.astype(np.float32),
        r_y_std.astype(np.float32),
        r_ex_std.astype(np.float32),
        aux,
    )


def model_to_raw_coords(
    r_x_std: np.ndarray,
    r_y_std: np.ndarray,
    r_ex_std: np.ndarray,
    *,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    state: BoundedLogitZscoreState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_x = r_x_std * state.std_r_x + state.mean_r_x
    r_y = r_y_std * state.std_r_y + state.mean_r_y
    r_ex = r_ex_std * state.std_r_ex + state.mean_r_ex
    s_x = sigmoid(r_x)
    s_y = sigmoid(r_y)
    s_ex = sigmoid(r_ex)
    x = x_lower + (x_upper - x_lower) * s_x
    e_y = s_y
    e_x = state.e_x_scale * s_ex
    return (
        np.atleast_1d(x).astype(np.float64),
        np.atleast_1d(e_y).astype(np.float64),
        np.atleast_1d(e_x).astype(np.float64),
    )


def jacobian_log_r_x_std_wrt_x(
    x: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    state: BoundedLogitZscoreState,
    *,
    s_x_clip: np.ndarray | None = None,
) -> np.ndarray:
    eps = state.eps
    span = np.maximum(x_upper - x_lower, eps)
    if s_x_clip is None:
        s_x = (x - x_lower) / span
        s_x_clip = np.clip(s_x, eps, 1.0 - eps)
    return (
        -_safe_log(np.array(state.std_r_x))
        - _safe_log(span)
        - _safe_log(s_x_clip)
        - _safe_log(1.0 - s_x_clip)
    )


def jacobian_log_r_y_std_wrt_e_y(
    e_y: np.ndarray,
    state: BoundedLogitZscoreState,
    *,
    s_y_clip: np.ndarray | None = None,
) -> np.ndarray:
    eps = state.eps
    if s_y_clip is None:
        s_y = np.clip(e_y, state.e_y_lower, state.e_y_upper)
        s_y_clip = np.clip(s_y, eps, 1.0 - eps)
    return (
        -_safe_log(np.array(state.std_r_y))
        - _safe_log(s_y_clip)
        - _safe_log(1.0 - s_y_clip)
    )


def jacobian_log_r_ex_std_wrt_e_x(
    e_x: np.ndarray,
    state: BoundedLogitZscoreState,
    *,
    s_ex_clip: np.ndarray | None = None,
) -> np.ndarray:
    eps = state.eps
    if s_ex_clip is None:
        s_ex = e_x / state.e_x_scale
        s_ex_clip = np.clip(s_ex, eps, 1.0 - eps)
    return (
        -_safe_log(np.array(state.std_r_ex))
        - _safe_log(np.array(state.e_x_scale))
        - _safe_log(s_ex_clip)
        - _safe_log(1.0 - s_ex_clip)
    )


def fit_bounded_logit_zscore_state(
    train_df,
    *,
    target_transform_cfg: dict[str, Any],
    bat_col: str = "bat_length_in",
) -> tuple[BoundedLogitZscoreState, dict[str, Any]]:
    """Fit global logit z-score stats from train split only."""
    cfg = target_transform_cfg
    eps = float(cfg.get("e_y_star", {}).get("eps", cfg.get("x", {}).get("eps", 1e-5)))
    x_cfg = cfg.get("x", {})
    ey_cfg = cfg.get("e_y_star", {})
    ex_cfg = cfg.get("e_x", {})

    x = pd.to_numeric(train_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    ey = pd.to_numeric(train_df["e_y_star"], errors="coerce").to_numpy(dtype=np.float64)
    ex = pd.to_numeric(train_df["e_x"], errors="coerce").to_numpy(dtype=np.float64)
    bat = pd.to_numeric(train_df[bat_col], errors="coerce").to_numpy(dtype=np.float64)

    x_lo, x_hi, emp_fb, x_mode = compute_x_bounds_per_row(bat, cfg_x=x_cfg, train_x=x)

    ey_lo = float(ey_cfg.get("lower", 0.0))
    ey_hi = float(ey_cfg.get("upper", 1.0))
    ex_lo = float(ex_cfg.get("lower", 0.0))
    ex_hi = float(ex_cfg.get("upper", 0.6))
    ex_scale = ex_hi

    eps_fit = float(x_cfg.get("eps", 1e-5))
    span_x = np.maximum(x_hi - x_lo, eps_fit)
    s_x = (x - x_lo) / span_x
    s_x_clip = np.clip(s_x, eps_fit, 1.0 - eps_fit)
    s_y_clip = np.clip(np.clip(ey, ey_lo, ey_hi), eps_fit, 1.0 - eps_fit)
    s_ex_clip = np.clip(np.clip(ex / ex_scale, 0.0, 1.0), eps_fit, 1.0 - eps_fit)
    r_x = logit(s_x_clip, eps_fit)
    r_y = logit(s_y_clip, eps_fit)
    r_ex = logit(s_ex_clip, eps_fit)

    def _ms(v: np.ndarray) -> tuple[float, float]:
        m = float(np.nanmean(v))
        s = float(np.nanstd(v))
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        return m, s

    mx, sx = _ms(r_x)
    my, sy = _ms(r_y)
    mex, sex = _ms(r_ex)

    state = BoundedLogitZscoreState(
        mx,
        sx,
        my,
        sy,
        mex,
        sex,
        eps_fit,
        x_mode,
        emp_fb,
        ey_lo,
        ey_hi,
        ex_lo,
        ex_hi,
        ex_scale,
    )

    manifest = {
        "target_transform": cfg,
        "state": state.to_dict(),
        "warnings": [],
    }
    if emp_fb:
        manifest["warnings"].append("empirical x support used for some/all rows (physics span invalid)")

    return state, manifest
