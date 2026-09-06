"""Bounded logit + z-score transforms for stage-u (v_ss_tilde, a_tilde)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

VA_NAMES = ("v_ss_tilde", "a_tilde")


@dataclass
class BoundedLogitZScoreTransform:
    """
    Map raw y_j in [L_j, U_j] to model space via logit + train z-score.

    ytilde_j = (logit(clamp((y_j-L)/(U-L), eps, 1-eps)) - mean_r_j) / std_r_j
    """

    bounds: dict[str, tuple[float, float]]
    eps: float = 1e-5
    stats: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in VA_NAMES:
            if name not in self.bounds:
                raise ValueError(f"bounds missing {name!r}")
        if self.stats is None:
            self.stats = {
                "mean_r": {n: 0.0 for n in VA_NAMES},
                "std_r": {n: 1.0 for n in VA_NAMES},
                "fit_counts": {},
                "out_of_bounds": {},
            }

    @property
    def L(self) -> np.ndarray:
        return np.array([self.bounds[n][0] for n in VA_NAMES], dtype=np.float64)

    @property
    def U(self) -> np.ndarray:
        return np.array([self.bounds[n][1] for n in VA_NAMES], dtype=np.float64)

    def _span(self) -> np.ndarray:
        return self.U - self.L

    def fit(
        self,
        raw_va: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit train means/stds of logit(r) on raw_va shape (N, 2)."""
        if raw_va.shape[1] != 2:
            raise ValueError(f"raw_va must be (N, 2), got {raw_va.shape}")
        r = self._raw_to_r(raw_va, track_oob=False)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            w = w / max(w.sum(), 1e-12)
            mean_r = {}
            std_r = {}
            for j, name in enumerate(VA_NAMES):
                v = r[:, j]
                m = float(np.average(v, weights=w))
                var = float(np.average((v - m) ** 2, weights=w))
                sig = float(np.sqrt(max(var, 1e-12)))
                mean_r[name] = m
                std_r[name] = sig
        else:
            mean_r = {name: float(r[:, j].mean()) for j, name in enumerate(VA_NAMES)}
            std_r = {
                name: float(max(r[:, j].std(ddof=0), 1e-8)) for j, name in enumerate(VA_NAMES)
            }
        self.stats = {
            "mean_r": mean_r,
            "std_r": std_r,
            "fit_counts": {"n_rows": int(raw_va.shape[0])},
            "out_of_bounds": self.stats.get("out_of_bounds", {}) if self.stats else {},
        }

    def _raw_to_s(self, raw_va: np.ndarray) -> np.ndarray:
        span = self._span()
        return (raw_va - self.L) / span

    def _raw_to_r(self, raw_va: np.ndarray, *, track_oob: bool = False) -> np.ndarray:
        s = self._raw_to_s(raw_va)
        if track_oob:
            n = raw_va.shape[0]
            oob = {}
            for j, name in enumerate(VA_NAMES):
                below = int(np.sum(raw_va[:, j] < self.L[j]))
                above = int(np.sum(raw_va[:, j] > self.U[j]))
                oob[name] = {
                    "below_L": below,
                    "above_U": above,
                    "pct_below_L": 100.0 * below / max(n, 1),
                    "pct_above_U": 100.0 * above / max(n, 1),
                }
            if self.stats is not None:
                self.stats["out_of_bounds"] = oob
        s_clipped = np.clip(s, self.eps, 1.0 - self.eps)
        return np.log(s_clipped / (1.0 - s_clipped))

    def transform(self, raw_va: np.ndarray) -> np.ndarray:
        r = self._raw_to_r(raw_va, track_oob=False)
        out = np.empty_like(r, dtype=np.float64)
        for j, name in enumerate(VA_NAMES):
            mu = float(self.stats["mean_r"][name])
            sig = float(max(self.stats["std_r"][name], 1e-8))
            out[:, j] = (r[:, j] - mu) / sig
        return out.astype(np.float32)

    def inverse(self, ytilde_va: np.ndarray) -> np.ndarray:
        """Map model-space (N,2) back to raw bounded (N,2)."""
        yt = np.asarray(ytilde_va, dtype=np.float64)
        r = np.empty_like(yt)
        for j, name in enumerate(VA_NAMES):
            mu = float(self.stats["mean_r"][name])
            sig = float(max(self.stats["std_r"][name], 1e-8))
            r[:, j] = yt[:, j] * sig + mu
        s = 1.0 / (1.0 + np.exp(-r))
        raw = self.L + s * self._span()
        return raw.astype(np.float32)

    def log_abs_det_dytilde_dy(self, raw_va: np.ndarray) -> np.ndarray:
        """
        Per-row log |d ytilde / d y| summed over v_ss_tilde and a_tilde.

        = -log(std_j) - log(U_j-L_j) - log(s_j) - log(1-s_j)
        with s_j the clipped proportion in (0,1).
        """
        raw = np.asarray(raw_va, dtype=np.float64)
        span = self._span()
        s = (raw - self.L) / span
        s_clipped = np.clip(s, self.eps, 1.0 - self.eps)
        log_det = np.zeros(raw.shape[0], dtype=np.float64)
        for j, name in enumerate(VA_NAMES):
            sig = float(max(self.stats["std_r"][name], 1e-8))
            log_det += (
                -np.log(sig)
                - np.log(span[j])
                - np.log(s_clipped[:, j])
                - np.log(1.0 - s_clipped[:, j])
            )
        return log_det.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "transform_type": "bounded_logit_zscore",
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "eps": self.eps,
            "stats": self.stats,
            "va_names": list(VA_NAMES),
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> BoundedLogitZScoreTransform:
        bounds = {k: (float(v[0]), float(v[1])) for k, v in d["bounds"].items()}
        return cls(bounds=bounds, eps=float(d.get("eps", 1e-5)), stats=d.get("stats"))


def quantile_report(values: np.ndarray, qs: tuple[float, ...] = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{int(q*1000)/10}": float("nan") for q in qs}
    out: dict[str, float] = {"min": float(np.min(v)), "max": float(np.max(v))}
    for q in qs:
        key = f"p{int(q * 1000) / 10}" if q < 0.1 else (f"p{int(q*100)}" if q >= 0.1 else f"p{q}")
        if q == 0.001:
            key = "p0.1"
        elif q == 0.01:
            key = "p1"
        elif q == 0.05:
            key = "p5"
        elif q == 0.5:
            key = "p50"
        elif q == 0.95:
            key = "p95"
        elif q == 0.99:
            key = "p99"
        elif q == 0.999:
            key = "p99.9"
        out[key] = float(np.quantile(v, q))
    return out


def bounds_diagnostics_for_split(
    df,
    bounds: dict[str, tuple[float, float]],
    split_name: str,
    *,
    eps: float = 1e-5,
) -> dict[str, Any]:
    """Summary stats for v_ss_tilde and a_tilde vs configured bounds."""
    diag: dict[str, Any] = {"split": split_name, "targets": {}}
    n = len(df)
    for name in VA_NAMES:
        if name not in df.columns:
            raise ValueError(f"{split_name}: missing {name}")
        v = pd_to_numeric_col(df[name])
        L, U = bounds[name]
        below = int(np.sum(v < L))
        above = int(np.sum(v > U))
        diag["targets"][name] = {
            "bounds": [L, U],
            "n_rows": n,
            "below_L": below,
            "above_U": above,
            "pct_below_L": 100.0 * below / max(n, 1),
            "pct_above_U": 100.0 * above / max(n, 1),
            "quantiles": quantile_report(v),
        }
    return diag


def pd_to_numeric_col(series) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


class PlayerSpecificBoundedLogitZScoreTransform:
    """
    Player-specific [L_ij, U_ij] logit bounds; global train z-score on r_j.
    """

    def __init__(
        self,
        player_bounds: Any,
        eps: float = 1e-5,
        stats: dict[str, Any] | None = None,
    ):
        from .player_target_bounds_u import PlayerTargetBounds

        if not isinstance(player_bounds, PlayerTargetBounds):
            raise TypeError("player_bounds must be PlayerTargetBounds")
        self.player_bounds = player_bounds
        self.eps = float(eps)
        if stats is None:
            self.stats = {
                "mean_r": {n: 0.0 for n in VA_NAMES},
                "std_r": {n: 1.0 for n in VA_NAMES},
            }
        else:
            self.stats = stats

    def fit(
        self,
        raw_df_train,
        player_ids: np.ndarray,
        *,
        sample_weight_col: str = "combined_training_weight",
    ) -> None:
        import pandas as pd

        w = None
        if sample_weight_col in raw_df_train.columns:
            w = pd.to_numeric(raw_df_train[sample_weight_col], errors="coerce").to_numpy(dtype=np.float64)
        raw_va = np.stack(
            [
                pd.to_numeric(raw_df_train[n], errors="coerce").to_numpy(dtype=np.float64)
                for n in VA_NAMES
            ],
            axis=1,
        )
        r_all = []
        for j in range(2):
            L, U = self.player_bounds.va_lower_upper(player_ids, j)
            span = (U - L).clip(min=1e-8)
            s = (raw_va[:, j] - L) / span
            s = np.clip(s, self.eps, 1.0 - self.eps)
            r_all.append(np.log(s / (1.0 - s)))
        r = np.column_stack(r_all)
        if w is not None:
            w = w / max(w.sum(), 1e-12)
            mean_r = {}
            std_r = {}
            for j, name in enumerate(VA_NAMES):
                v = r[:, j]
                m = float(np.average(v, weights=w))
                var = float(np.average((v - m) ** 2, weights=w))
                std_r[name] = float(np.sqrt(max(var, 1e-12)))
                mean_r[name] = m
        else:
            mean_r = {name: float(r[:, j].mean()) for j, name in enumerate(VA_NAMES)}
            std_r = {name: float(max(r[:, j].std(ddof=0), 1e-8)) for j, name in enumerate(VA_NAMES)}
        self.stats = {"mean_r": mean_r, "std_r": std_r, "n_rows_fit": int(len(raw_df_train))}

    def transform(self, raw_va: np.ndarray, player_ids: np.ndarray) -> np.ndarray:
        raw_va = np.asarray(raw_va, dtype=np.float64)
        r = np.empty_like(raw_va)
        for j in range(2):
            L, U = self.player_bounds.va_lower_upper(player_ids, j)
            span = (U - L).clip(min=1e-8)
            s = np.clip((raw_va[:, j] - L) / span, self.eps, 1.0 - self.eps)
            r[:, j] = np.log(s / (1.0 - s))
        out = np.empty_like(r)
        for j, name in enumerate(VA_NAMES):
            mu = float(self.stats["mean_r"][name])
            sig = float(max(self.stats["std_r"][name], 1e-8))
            out[:, j] = (r[:, j] - mu) / sig
        return out.astype(np.float32)

    def inverse(self, ytilde_va: np.ndarray, player_ids: np.ndarray) -> np.ndarray:
        yt = np.asarray(ytilde_va, dtype=np.float64)
        raw = np.empty_like(yt)
        for j, name in enumerate(VA_NAMES):
            mu = float(self.stats["mean_r"][name])
            sig = float(max(self.stats["std_r"][name], 1e-8))
            r = yt[:, j] * sig + mu
            s = 1.0 / (1.0 + np.exp(-r))
            L, U = self.player_bounds.va_lower_upper(player_ids, j)
            raw[:, j] = L + (U - L) * s
        return raw.astype(np.float32)

    def log_abs_det_dytilde_dy(self, raw_va: np.ndarray, player_ids: np.ndarray) -> np.ndarray:
        raw_va = np.asarray(raw_va, dtype=np.float64)
        log_det = np.zeros(raw_va.shape[0], dtype=np.float64)
        for j, name in enumerate(VA_NAMES):
            L, U = self.player_bounds.va_lower_upper(player_ids, j)
            span = (U - L).clip(min=1e-8)
            s = np.clip((raw_va[:, j] - L) / span, self.eps, 1.0 - self.eps)
            sig = float(max(self.stats["std_r"][name], 1e-8))
            log_det += (
                -np.log(sig)
                - np.log(span)
                - np.log(s)
                - np.log(1.0 - s)
            )
        return log_det.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "transform_type": "player_specific_bounded_logit_zscore",
            "eps": self.eps,
            "stats": self.stats,
            "player_bounds": self.player_bounds.state_dict(),
            "va_names": list(VA_NAMES),
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> PlayerSpecificBoundedLogitZScoreTransform:
        from .player_target_bounds_u import PlayerTargetBounds

        pb = PlayerTargetBounds.from_state_dict(d["player_bounds"])
        return cls(player_bounds=pb, eps=float(d.get("eps", 1e-5)), stats=d.get("stats"))


def uses_player_specific_bounded_transform(cfg: dict[str, Any]) -> bool:
    return (
        cfg.get("target_transform", {}).get("va_transform")
        == "player_specific_bounded_logit_zscore"
    )


def warn_if_train_oob_exceeds(
    train_diag: dict[str, Any],
    threshold_pct: float = 0.5,
) -> list[str]:
    warnings: list[str] = []
    for name, block in train_diag.get("targets", {}).items():
        pct = block["pct_below_L"] + block["pct_above_U"]
        if pct > threshold_pct:
            warnings.append(
                f"TRAINING WARNING: {name!r} has {pct:.3f}% rows outside "
                f"[{block['bounds'][0]}, {block['bounds'][1]}] (threshold {threshold_pct}%)"
            )
    return warnings
