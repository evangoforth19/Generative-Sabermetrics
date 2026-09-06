"""Post-hoc EV calibration for physics decoder outputs (does not retrain stage-u/z)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def soft_cap_ev(y: np.ndarray | float, *, cap: float = 121.0, tau: float = 2.0) -> np.ndarray:
    """
    Smooth soft cap: cap - tau * log(1 + exp((cap - y) / tau)).
    Numerically stable via logaddexp: log(1 + exp(x)) = logaddexp(0, x).
    """
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    x = (cap - ya) / tau
    log1pe = np.logaddexp(0.0, x)
    out = cap - tau * log1pe
    return out


def summarize_ev_per_event(draws: pd.DataFrame, ev_col: str, group_col: str = "event_id") -> pd.DataFrame:
    """Aggregate decoder EV draws into event-level statistics."""
    g = draws.groupby(group_col, sort=False)[ev_col]
    agg = g.agg(
        EV_dec_mean="mean",
        EV_dec_std=lambda s: float(np.std(s, ddof=0)),
        EV_dec_p05=lambda s: float(np.quantile(s, 0.05)),
        EV_dec_p10=lambda s: float(np.quantile(s, 0.10)),
        EV_dec_p25=lambda s: float(np.quantile(s, 0.25)),
        EV_dec_p50=lambda s: float(np.quantile(s, 0.50)),
        EV_dec_p75=lambda s: float(np.quantile(s, 0.75)),
        EV_dec_p90=lambda s: float(np.quantile(s, 0.90)),
        EV_dec_p95=lambda s: float(np.quantile(s, 0.95)),
        EV_dec_p99=lambda s: float(np.quantile(s, 0.99)),
        n_draws="count",
    ).reset_index()
    return agg


CONTEXT_COLUMNS = [
    "pitch_group",
    "pitch_type",
    "z_count",
    "release_speed",
    "spin_rate",
    "release_spin_rate",
    "spin_axis",
    "plate_x",
    "plate_z",
    "batter_hand",
    "stand",
    "pitcher_hand",
    "p_throws",
    "balls",
    "strikes",
]


@dataclass
class EVPhysicsCalibrator:
    """
    Residual correction: EV_resid = EV_obs - EV_dec_mean ~ features.
    Per-draw: EV_cal = soft_cap(EV_cal_mean + rho * (EV_dec - EV_dec_mean)).
    """

    rho: float = 0.85
    cap: float = 121.0
    tau: float = 2.0
    ev_col: str = "EV"
    group_col: str = "event_id"
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    pipeline_: Pipeline | None = None
    target_name: str = "EV_resid"

    def fit(self, event_summary_df: pd.DataFrame) -> EVPhysicsCalibrator:
        """Fit residual model. ``event_summary_df`` must include EV_resid and all feature columns."""
        self._build_pipeline()
        feats = self.numeric_features + self.categorical_features
        X = event_summary_df[feats].copy()
        y = event_summary_df[self.target_name].to_numpy(dtype=np.float64)
        self.pipeline_.fit(X, y)
        return self

    def _build_pipeline(self) -> None:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers: list[tuple[str, Pipeline, list[str]]] = [
            ("num", num_pipe, self.numeric_features),
        ]
        if self.categorical_features:
            try:
                oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
            except TypeError:
                try:
                    oh = OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=30)
                except TypeError:
                    oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh", oh),
                ]
            )
            transformers.append(("cat", cat_pipe, self.categorical_features))
        prep = ColumnTransformer(transformers, remainder="drop")
        reg = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=8,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        )
        self.pipeline_ = Pipeline([("prep", prep), ("reg", reg)])

    def predict_residual(self, event_summary_df: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Calibrator not fitted")
        X = event_summary_df[self.numeric_features + self.categorical_features].copy()
        return np.asarray(self.pipeline_.predict(X), dtype=np.float64)

    def predict_event_mean(self, event_summary_df: pd.DataFrame) -> np.ndarray:
        """EV_cal_mean = EV_dec_mean + predicted_residual."""
        resid = self.predict_residual(event_summary_df)
        mu = event_summary_df["EV_dec_mean"].to_numpy(dtype=np.float64)
        return mu + resid

    def transform_draws(self, draws_df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
        """
        Add EV_cal per draw. Recomputes event summaries from draws for consistency.
        Preserves original EV column; adds EV_cal (and EV_dec alias if EV renamed).
        """
        gc = group_col or self.group_col
        evc = self.ev_col
        if evc not in draws_df.columns:
            raise ValueError(f"Draws missing EV column {evc!r}; have {list(draws_df.columns)[:40]}")

        out = draws_df.copy()
        out["_EV_dec"] = pd.to_numeric(out[evc], errors="coerce")

        summ = summarize_ev_per_event(out.dropna(subset=["_EV_dec"]), "_EV_dec", group_col=gc)
        feat_cols = self.numeric_features + self.categorical_features
        for c in feat_cols:
            if c in summ.columns:
                continue
            if c in out.columns:
                tmp = out.groupby(gc, sort=False)[c].first().reset_index()
                summ = summ.merge(tmp, on=gc, how="left")
            else:
                summ[c] = np.nan
        for c in feat_cols:
            if c not in summ.columns:
                summ[c] = np.nan

        summ["_pred_resid"] = self.predict_residual(summ)
        summ["EV_cal_mean"] = summ["EV_dec_mean"].to_numpy(dtype=np.float64) + summ["_pred_resid"].to_numpy(
            dtype=np.float64
        )
        mean_map = summ.set_index(gc)["EV_cal_mean"]
        mu_dec_map = summ.set_index(gc)["EV_dec_mean"]

        cal_mean = out[gc].map(mean_map)
        dec_mean = out[gc].map(mu_dec_map)
        ev_temp = cal_mean.to_numpy(dtype=np.float64) + self.rho * (
            out["_EV_dec"].to_numpy(dtype=np.float64) - dec_mean.to_numpy(dtype=np.float64)
        )
        ev_cal = soft_cap_ev(ev_temp, cap=self.cap, tau=self.tau)
        out["EV_cal"] = ev_cal
        out["EV_dec_mean_merged"] = dec_mean.to_numpy(dtype=np.float64)
        out["EV_cal_mean_merged"] = cal_mean.to_numpy(dtype=np.float64)
        out.drop(columns=["_EV_dec"], errors="ignore", inplace=True)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rho": self.rho,
            "cap": self.cap,
            "tau": self.tau,
            "ev_col": self.ev_col,
            "group_col": self.group_col,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_name": self.target_name,
            "pipeline": self.pipeline_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> EVPhysicsCalibrator:
        path = Path(path)
        payload = joblib.load(path)
        obj = cls(
            rho=float(payload["rho"]),
            cap=float(payload["cap"]),
            tau=float(payload["tau"]),
            ev_col=str(payload["ev_col"]),
            group_col=str(payload["group_col"]),
            numeric_features=list(payload["numeric_features"]),
            categorical_features=list(payload["categorical_features"]),
        )
        obj.target_name = str(payload.get("target_name", "EV_resid"))
        obj.pipeline_ = payload["pipeline"]
        return obj

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "rho": self.rho,
            "cap": self.cap,
            "tau": self.tau,
            "ev_col": self.ev_col,
            "group_col": self.group_col,
        }


def merge_context_on_events(
    event_df: pd.DataFrame,
    master: pd.DataFrame,
    event_key: str = "event_id",
) -> pd.DataFrame:
    """Left-merge context columns from master onto event-level table."""
    cols = [event_key] + [c for c in CONTEXT_COLUMNS if c in master.columns]
    cols = list(dict.fromkeys(cols))
    sub = master[cols].drop_duplicates(subset=[event_key], keep="first")
    return event_df.merge(sub, on=event_key, how="left")
