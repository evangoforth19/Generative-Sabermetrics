"""Stage-z internal target transforms (raw contract columns -> training coordinates)."""

from __future__ import annotations

from typing import Any

import numpy as np

# Raw parquet / contract order
RAW_FOUR_ORDER = ("x", "psi_deg", "e_y_star", "theta_deg")

# Internal circular training order (6D)
CIRCULAR_SIX_ORDER = ("x", "e_y_star", "sin_psi", "cos_psi", "sin_theta", "cos_theta")


def wrap_deg(delta: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle difference(s) to (-180, 180]."""
    x = np.asarray(delta, dtype=np.float64)
    w = (x + 180.0) % 360.0 - 180.0
    w = np.where(w <= -180.0, 180.0, w)
    return w if np.ndim(delta) else float(w.flat[0])


def raw_four_to_circ_six(y_raw_four: np.ndarray) -> np.ndarray:
    """
    y_raw_four: (N, 4) columns x, psi_deg, e_y_star, theta_deg.
    Returns (N, 6): x, e_y_star, sin_psi, cos_psi, sin_theta, cos_theta (angles in rad for sin/cos).
    """
    if y_raw_four.ndim != 2 or y_raw_four.shape[1] != 4:
        raise ValueError(f"expected (N, 4), got {y_raw_four.shape}")
    x = y_raw_four[:, 0].astype(np.float64)
    psi = y_raw_four[:, 1].astype(np.float64)
    ey = y_raw_four[:, 2].astype(np.float64)
    th = y_raw_four[:, 3].astype(np.float64)
    pr = np.radians(psi)
    tr = np.radians(th)
    return np.stack([x, ey, np.sin(pr), np.cos(pr), np.sin(tr), np.cos(tr)], axis=1)


def circ_six_unstandardize(y_std: np.ndarray, circ_means: np.ndarray, circ_stds: np.ndarray) -> np.ndarray:
    return y_std * circ_stds + circ_means


def angles_deg_from_circ_six(y_nat_circ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    y_nat_circ: (..., 6) natural units after un-standardizing [x, ey, sp, cp, st, ct].
    Returns psi_deg, theta_deg with shape (...,).
    """
    sp = y_nat_circ[..., 2]
    cp = y_nat_circ[..., 3]
    st = y_nat_circ[..., 4]
    ct = y_nat_circ[..., 5]
    psi = np.degrees(np.arctan2(sp, cp))
    th = np.degrees(np.arctan2(st, ct))
    return psi, th


def mixture_mean_direction_deg(sin_vals: np.ndarray, cos_vals: np.ndarray) -> np.ndarray:
    """Point estimate atan2(E[sin], E[cos]) in degrees, shape (...) from leading dims."""
    return np.degrees(np.arctan2(sin_vals, cos_vals))


def circular_target_transform_manifest(
    circ_means: np.ndarray,
    circ_stds: np.ndarray,
    *,
    raw_order: tuple[str, ...] = RAW_FOUR_ORDER,
    circ_order: tuple[str, ...] = CIRCULAR_SIX_ORDER,
) -> dict[str, Any]:
    return {
        "parameterization": "circular_6d",
        "raw_targets_contract_order": list(raw_order),
        "internal_training_order": list(circ_order),
        "standardization": "Train-only mean/std on each of the 6 internal coordinates after sin/cos; "
        "x and e_y_star use the same mean/std as manifests/standardization_stats.json for those raw columns.",
        "circ_means": circ_means.tolist(),
        "circ_stds": circ_stds.tolist(),
        "angles_in_radians_before_trig": True,
    }


def wrapped_abs_error_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    return np.abs(wrap_deg(pred_deg - true_deg))


def weighted_wrapped_mae_rmse(
    pred_deg: np.ndarray,
    true_deg: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float]:
    d = wrap_deg(pred_deg - true_deg)
    w = np.asarray(w, dtype=np.float64)
    sw = float(np.sum(w).clip(min=1e-12))
    mae = float(np.sum(np.abs(d) * w) / sw)
    rmse = float(np.sqrt(np.sum((d**2) * w) / sw))
    return mae, rmse
