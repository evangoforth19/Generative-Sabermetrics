"""
Load and apply the calibrated LightGBM batted-ball outcome surrogate:
(EV, LA, SA) -> P(out, single, double, triple, home_run).

Expects artifacts in ``model_dir`` from ``train_batted_ball_value_surface.py``:
``model_lgbm.pkl``, ``label_encoder.pkl`` (dict ``class_to_idx`` / ``idx_to_class``),
``calibration_temperature.json``, ``model_metadata.json``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HOME_X = 125.42
HOME_Y = 198.27

CLASS_ORDER = ["out", "single", "double", "triple", "home_run"]

DEFAULT_XWOBA_WEIGHTS = np.array([0.0, 0.902, 1.279, 1.618, 2.078], dtype=np.float64)

FEATURE_COLS = ["launch_speed", "launch_angle", "sin_spray_angle", "cos_spray_angle"]


def _to_arrays(
    ev: Any, la: Any, sa: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    eva = np.asarray(ev, dtype=np.float64)
    laa = np.asarray(la, dtype=np.float64)
    saa = np.asarray(sa, dtype=np.float64)
    scalar = eva.ndim == 0 and laa.ndim == 0 and saa.ndim == 0
    eva = np.atleast_1d(eva).reshape(-1)
    laa = np.atleast_1d(laa).reshape(-1)
    saa = np.atleast_1d(saa).reshape(-1)
    if not (len(eva) == len(laa) == len(saa)):
        raise ValueError("EV, LA, SA must have compatible shapes (scalars or same-length arrays).")
    return eva, laa, saa, scalar


def _spray_trig(sa_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.radians(sa_deg)
    return np.sin(rad), np.cos(rad)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def load_batted_ball_value_model(model_dir: str | Path) -> dict[str, Any]:
    """Load model bundle from ``model_dir``."""
    d = Path(model_dir).resolve()
    with open(d / "calibration_temperature.json", encoding="utf-8") as f:
        temp = json.load(f)
    T = float(temp["temperature"])
    with open(d / "model_lgbm.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(d / "label_encoder.pkl", "rb") as f:
        enc = pickle.load(f)  # dict with class_to_idx, idx_to_class (semantic CLASS_ORDER)
    meta_path = d / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "model_dir": d,
        "classifier": clf,
        "label_encoding": enc,
        "temperature": T,
        "metadata": meta,
        "class_order": list(meta.get("class_order", CLASS_ORDER)),
    }


def _build_X(ev: np.ndarray, la: np.ndarray, sa: np.ndarray) -> pd.DataFrame:
    s, c = _spray_trig(sa)
    return pd.DataFrame(
        np.column_stack([ev, la, s, c]),
        columns=FEATURE_COLS,
    )


def predict_outcome_probs(
    ev: Any,
    la: Any,
    sa: Any,
    model_bundle: dict[str, Any],
    *,
    calibrated: bool = True,
) -> np.ndarray:
    """
    Return shape (n, 5) probabilities in order
    ``['out', 'single', 'double', 'triple', 'home_run']``.
    """
    clf = model_bundle["classifier"]
    T = float(model_bundle["temperature"])
    eva, laa, saa, scalar = _to_arrays(ev, la, sa)
    X = _build_X(eva, laa, saa)
    if calibrated and T > 0:
        raw = clf.predict(X, raw_score=True)
        probs = _softmax(raw / T)
    else:
        probs = clf.predict_proba(X)
    if scalar:
        return probs[0].astype(np.float64)
    return probs.astype(np.float64)


def predict_xwobacon3d(
    ev: Any,
    la: Any,
    sa: Any,
    model_bundle: dict[str, Any],
    weights: np.ndarray | None = None,
    *,
    calibrated: bool = True,
) -> np.ndarray | float:
    """Expected xwOBAcon from calibrated outcome probabilities (default Statcast-style weights)."""
    w = DEFAULT_XWOBA_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64).reshape(5)
    if w.shape != (5,):
        raise ValueError("weights must have shape (5,) matching class order.")
    probs = predict_outcome_probs(ev, la, sa, model_bundle, calibrated=calibrated)
    eva, _, _, scalar = _to_arrays(ev, la, sa)
    if scalar:
        return float(np.dot(probs, w))
    return (probs @ w).astype(np.float64)
