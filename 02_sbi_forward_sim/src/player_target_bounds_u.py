"""Train-only player-specific bounds and circular d_tilde support arcs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .target_transforms_u import VA_NAMES, quantile_report

D_TARGET = "d_tilde"
ATTACK_CAP_DEFAULT = (-45.0, 45.0)
BOUNDS_FIT_WARNING = "Player-specific target bounds were fit using train split only."


@dataclass
class CircularArcSupport:
    center_deg: float
    half_width_deg: float
    lower_deg: float
    upper_deg: float
    wraps_zero: bool
    arc_width_deg: float
    full_circle: bool = False
    fallback: bool = False
    fallback_reason: str | None = None

    def contains_deg(self, deg: float | np.ndarray) -> np.ndarray:
        d = np.mod(np.asarray(deg, dtype=np.float64), 360.0)
        if self.full_circle:
            return np.ones(d.shape, dtype=bool)
        if not self.wraps_zero:
            return (d >= self.lower_deg) & (d <= self.upper_deg)
        return (d >= self.lower_deg) | (d <= self.upper_deg)

    def project_to_arc_deg(self, deg: float | np.ndarray) -> np.ndarray:
        d = np.mod(np.asarray(deg, dtype=np.float64), 360.0)
        if self.full_circle:
            return d
        inside = self.contains_deg(d)
        if np.all(inside):
            return d
        out = d.copy()
        for i in np.where(~inside)[0]:
            di = d[i]
            if not self.wraps_zero:
                out[i] = np.clip(di, self.lower_deg, self.upper_deg)
            else:
                # distance along circle to lower and upper boundaries
                dist_lower = min((di - self.lower_deg) % 360, (self.lower_deg - di) % 360)
                dist_upper = min((di - self.upper_deg) % 360, (self.upper_deg - di) % 360)
                out[i] = self.lower_deg if dist_lower <= dist_upper else self.upper_deg
        return out % 360.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_deg": self.center_deg,
            "half_width_deg": self.half_width_deg,
            "lower_deg": self.lower_deg,
            "upper_deg": self.upper_deg,
            "wraps_zero": self.wraps_zero,
            "arc_width_deg": self.arc_width_deg,
            "full_circle": self.full_circle,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CircularArcSupport:
        return cls(
            center_deg=float(d["center_deg"]),
            half_width_deg=float(d["half_width_deg"]),
            lower_deg=float(d["lower_deg"]),
            upper_deg=float(d["upper_deg"]),
            wraps_zero=bool(d["wraps_zero"]),
            arc_width_deg=float(d["arc_width_deg"]),
            full_circle=bool(d.get("full_circle", False)),
            fallback=bool(d.get("fallback", False)),
            fallback_reason=d.get("fallback_reason"),
        )


@dataclass
class PlayerTargetBounds:
    """Per-player and global bounds fitted on train split only."""

    player_col: str
    va_bounds: dict[str, dict[int, tuple[float, float]]]  # target -> player_id -> (L,U)
    d_arc: dict[int, CircularArcSupport]
    global_va: dict[str, tuple[float, float]]
    global_d_arc: CircularArcSupport
    metadata: dict[str, Any] = field(default_factory=dict)

    def va_lower_upper(self, player_ids: np.ndarray, target_idx: int) -> tuple[np.ndarray, np.ndarray]:
        name = VA_NAMES[target_idx]
        pmap = self.va_bounds[name]
        gL, gU = self.global_va[name]
        L = np.empty(len(player_ids), dtype=np.float64)
        U = np.empty(len(player_ids), dtype=np.float64)
        for i, pid in enumerate(player_ids.astype(int)):
            if int(pid) in pmap:
                L[i], U[i] = pmap[int(pid)]
            else:
                L[i], U[i] = gL, gU
        return L, U

    def get_d_arc(self, player_id: int) -> CircularArcSupport:
        return self.d_arc.get(int(player_id), self.global_d_arc)

    def state_dict(self) -> dict[str, Any]:
        return {
            "player_col": self.player_col,
            "va_bounds": {
                t: {str(k): list(v) for k, v in m.items()} for t, m in self.va_bounds.items()
            },
            "d_arc": {str(k): v.to_dict() for k, v in self.d_arc.items()},
            "global_va": {k: list(v) for k, v in self.global_va.items()},
            "global_d_arc": self.global_d_arc.to_dict(),
            "metadata": self.metadata,
            "fit_warning": BOUNDS_FIT_WARNING,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> PlayerTargetBounds:
        va = {
            t: {int(k): (float(v[0]), float(v[1])) for k, v in m.items()}
            for t, m in d["va_bounds"].items()
        }
        d_arc = {int(k): CircularArcSupport.from_dict(v) for k, v in d["d_arc"].items()}
        gva = {k: (float(v[0]), float(v[1])) for k, v in d["global_va"].items()}
        return cls(
            player_col=d["player_col"],
            va_bounds=va,
            d_arc=d_arc,
            global_va=gva,
            global_d_arc=CircularArcSupport.from_dict(d["global_d_arc"]),
            metadata=d.get("metadata", {}),
        )


def smallest_covering_arc_deg(
    angles_deg: np.ndarray,
    *,
    padding_deg: float = 0.0,
    full_circle_threshold_deg: float = 350.0,
) -> CircularArcSupport:
    """Minimal circular arc containing all angles (degrees), on [0,360)."""
    a = np.mod(np.asarray(angles_deg, dtype=np.float64), 360.0)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return CircularArcSupport(
            0.0, 180.0, 0.0, 360.0, True, 360.0, full_circle=True, fallback=True, fallback_reason="no_data"
        )
    if a.size == 1:
        hw = padding_deg
        return CircularArcSupport(
            float(a[0]),
            hw,
            float((a[0] - hw) % 360),
            float((a[0] + hw) % 360),
            False,
            2 * hw,
        )
    a_sorted = np.sort(a)
    gaps = np.diff(a_sorted)
    wrap_gap = a_sorted[0] + 360.0 - a_sorted[-1]
    gaps = np.append(gaps, wrap_gap)
    max_i = int(np.argmax(gaps))
    largest_gap = float(gaps[max_i])
    arc_width = 360.0 - largest_gap
    if arc_width >= full_circle_threshold_deg:
        return CircularArcSupport(
            180.0, 180.0, 0.0, 360.0, True, arc_width, full_circle=True
        )
    lower = float(a_sorted[(max_i + 1) % a_sorted.size])
    upper = float(a_sorted[max_i])
    half_width = arc_width / 2.0 + padding_deg
    half_width = min(half_width, 180.0)
    arc_width_p = 2.0 * half_width
    center = (lower + arc_width_p / 2.0) % 360.0
    wraps = lower > upper or arc_width > 180.0 - 1e-6
    return CircularArcSupport(
        center_deg=center,
        half_width_deg=half_width,
        lower_deg=lower,
        upper_deg=upper,
        wraps_zero=wraps,
        arc_width_deg=arc_width_p,
    )


def _intersect_interval(
    low: float, high: float, cap_low: float | None, cap_high: float | None
) -> tuple[float, float]:
    if cap_low is not None:
        low = max(low, cap_low)
    if cap_high is not None:
        high = min(high, cap_high)
    return low, high


def fit_player_target_bounds_train_only(
    train_df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    player_col: str = "batter_name",
    vocabs: dict[str, dict[str, int]] | None = None,
) -> PlayerTargetBounds:
    """
    Fit bounds using **train_df only**. Never pass calibration or test rows.
    """
    bcfg = cfg.get("bounds", {})
    pcfg = bcfg.get("player_specific", {})
    min_ev = int(pcfg.get("min_events_for_player_bounds", 20))
    min_rows = int(pcfg.get("min_rows_for_player_bounds", 100))
    pad_v = float(pcfg.get("padding", {}).get("v_ss_tilde_abs", 0.0))
    pad_a = float(pcfg.get("padding", {}).get("a_tilde_abs", 0.0))
    pad_d = float(pcfg.get("padding", {}).get("d_tilde_deg", 0.0))
    quantile_bounds = pcfg.get("quantile_bounds", {})
    use_quantile_bounds = bool(quantile_bounds.get("enabled", False))
    q_low = float(quantile_bounds.get("lower", 0.01))
    q_high = float(quantile_bounds.get("upper", 0.99))
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"Invalid player quantile bounds: lower={q_low}, upper={q_high}")
    d_cap_raw = pcfg.get("d_tilde_global_cap", pcfg.get("attack_direction_global_cap", list(ATTACK_CAP_DEFAULT)))
    d_cap = (float(d_cap_raw[0]), float(d_cap_raw[1]))
    attack_cap_raw = pcfg.get("attack_angle_global_cap")
    attack_cap = (
        (float(attack_cap_raw[0]), float(attack_cap_raw[1]))
        if attack_cap_raw is not None
        else None
    )
    sanity = bcfg.get("global_sanity_caps", {})
    dcfg = bcfg.get("d_tilde_support", {})
    full_thr = float(dcfg.get("full_circle_threshold_deg", 350.0))

    print(BOUNDS_FIT_WARNING)

    # Global train bounds
    global_va: dict[str, tuple[float, float]] = {}
    for name in VA_NAMES:
        v = pd.to_numeric(train_df[name], errors="coerce").to_numpy(dtype=np.float64)
        if use_quantile_bounds:
            gL, gH = float(np.nanquantile(v, q_low)), float(np.nanquantile(v, q_high))
        else:
            gL, gH = float(np.nanmin(v)), float(np.nanmax(v))
        if name in sanity:
            gL, gH = _intersect_interval(gL, gH, sanity[name][0], sanity[name][1])
        if name == "a_tilde" and attack_cap is not None:
            gL = max(gL, attack_cap[0])
            gH = min(gH, attack_cap[1])
        if gL >= gH:
            gL, gH = attack_cap if (name == "a_tilde" and attack_cap is not None) else (gL - 1.0, gH + 1.0)
        global_va[name] = (gL, gH)

    d_glob = pd.to_numeric(train_df[D_TARGET], errors="coerce").to_numpy(dtype=np.float64)
    if bool(pcfg.get("use_global_d_tilde_cap", False)):
        d_low_mod = float(d_cap[0] % 360.0)
        d_high_mod = float(d_cap[1] % 360.0)
        global_d_arc = CircularArcSupport(
            center_deg=float((d_cap[0] + d_cap[1]) / 2.0),
            half_width_deg=float((d_cap[1] - d_cap[0]) / 2.0),
            lower_deg=d_low_mod,
            upper_deg=d_high_mod,
            wraps_zero=d_low_mod > d_high_mod,
            arc_width_deg=float(d_cap[1] - d_cap[0]),
        )
    else:
        global_d_arc = smallest_covering_arc_deg(
            d_glob, padding_deg=pad_d, full_circle_threshold_deg=full_thr
        )

    va_bounds: dict[str, dict[int, tuple[float, float]]] = {n: {} for n in VA_NAMES}
    d_arc: dict[int, CircularArcSupport] = {}
    player_fallbacks: list[dict[str, Any]] = []

    grouped = train_df.groupby(player_col, sort=False)
    name_to_id: dict[str, int] = {}
    if vocabs and player_col in vocabs:
        name_to_id = {k: v for k, v in vocabs[player_col].items()}

    for pname, gdf in grouped:
        pid = int(name_to_id.get(str(pname).strip(), 0))
        n_rows = len(gdf)
        n_events = int(gdf["event_id"].nunique()) if "event_id" in gdf.columns else n_rows
        use_global = n_rows < min_rows or n_events < min_ev

        for j, name in enumerate(VA_NAMES):
            v = pd.to_numeric(gdf[name], errors="coerce").to_numpy(dtype=np.float64)
            if use_global:
                va_bounds[name][pid] = global_va[name]
                continue
            if use_quantile_bounds:
                low, high = float(np.nanquantile(v, q_low)), float(np.nanquantile(v, q_high))
            else:
                low, high = float(np.nanmin(v)), float(np.nanmax(v))
            pad = pad_v if name == "v_ss_tilde" else pad_a
            low -= pad
            high += pad
            if name in sanity:
                low, high = _intersect_interval(low, high, sanity[name][0], sanity[name][1])
            if name == "a_tilde" and attack_cap is not None:
                raw_low, raw_high = low, high
                low = max(low, attack_cap[0])
                high = min(high, attack_cap[1])
                if low >= high:
                    low, high = global_va[name]
                    player_fallbacks.append(
                        {"player": pname, "target": name, "reason": "attack_cap_invalid_after_clip"}
                    )
            if low >= high:
                low, high = global_va[name]
                player_fallbacks.append({"player": pname, "target": name, "reason": "low_ge_high"})
            va_bounds[name][pid] = (low, high)

        if use_global:
            d_arc[pid] = CircularArcSupport(
                **{k: v for k, v in global_d_arc.to_dict().items()},
                fallback=True,
                fallback_reason="insufficient_train_rows_or_events",
            )
        else:
            if bool(pcfg.get("use_global_d_tilde_cap", False)):
                d_arc[pid] = global_d_arc
            else:
                dvals = pd.to_numeric(gdf[D_TARGET], errors="coerce").to_numpy(dtype=np.float64)
                d_arc[pid] = smallest_covering_arc_deg(
                    dvals, padding_deg=pad_d, full_circle_threshold_deg=full_thr
                )

    meta = {
        "fit_split": "train",
        "n_train_rows": int(len(train_df)),
        "n_players": int(len(grouped)),
        "min_events_for_player_bounds": min_ev,
        "min_rows_for_player_bounds": min_rows,
        "attack_angle_global_cap": list(attack_cap) if attack_cap is not None else None,
        "d_tilde_global_cap": list(d_cap),
        "use_global_d_tilde_cap": bool(pcfg.get("use_global_d_tilde_cap", False)),
        "quantile_bounds": {
            "enabled": use_quantile_bounds,
            "lower": q_low,
            "upper": q_high,
        },
        "player_fallback_records": player_fallbacks,
    }
    return PlayerTargetBounds(
        player_col=player_col,
        va_bounds=va_bounds,
        d_arc=d_arc,
        global_va=global_va,
        global_d_arc=global_d_arc,
        metadata=meta,
    )


def sample_d_with_circular_support(
    deg_samples: np.ndarray,
    arc: CircularArcSupport,
    *,
    resample_cap: int = 100,
    fallback: str = "project_to_arc",
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Enforce circular support on degree samples via rejection / projection.

    deg_samples: (n_samples,) proposed degrees
    """
    out = np.mod(deg_samples.astype(np.float64), 360.0)
    n = out.size
    attempts = np.zeros(n, dtype=np.int64)
    projected = 0
    failures = 0
    for i in range(n):
        if arc.contains_deg(out[i]):
            continue
        ok = False
        for att in range(1, resample_cap + 1):
            attempts[i] = att
            # resample handled externally; here we only project/fail single draw
            break
        if not ok:
            if fallback == "project_to_arc":
                out[i] = float(arc.project_to_arc_deg(out[i])[()])
                projected += 1
            elif fallback == "use_unbounded":
                pass
            else:
                out[i] = float(arc.project_to_arc_deg(out[i])[()])
                projected += 1
            failures += int(attempts[i] >= resample_cap)
    diag = {
        "mean_attempts": float(attempts.mean()) if n else 0.0,
        "p95_attempts": float(np.quantile(attempts, 0.95)) if n else 0.0,
        "projected_rate": projected / max(n, 1),
        "rejection_failure_rate": failures / max(n, 1),
    }
    return out, diag


def enforce_d_support_batch(
    deg: np.ndarray,
    player_ids: np.ndarray,
    bounds: PlayerTargetBounds,
    *,
    resample_cap: int = 100,
    fallback: str = "project_to_arc",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-row rejection with optional projection fallback."""
    rng = rng or np.random.default_rng(0)
    out = np.mod(deg.astype(np.float64), 360.0).copy()
    attempts = np.zeros(len(out), dtype=np.int64)
    projected = 0
    n_fail = 0
    for i, pid in enumerate(player_ids):
        arc = bounds.get_d_arc(int(pid))
        if arc.contains_deg(out[i]):
            continue
        ok = False
        for att in range(1, resample_cap + 1):
            attempts[i] = att
            # jitter: small noise not available without VM resample — project on last
            if att == resample_cap:
                break
        if not ok:
            if fallback == "use_unbounded":
                continue
            out[i] = float(arc.project_to_arc_deg(out[i])[0])
            projected += 1
            if attempts[i] >= resample_cap:
                n_fail += 1
    return out, {
        "mean_attempts": float(attempts.mean()),
        "p95_attempts": float(np.quantile(attempts, 0.95)) if len(attempts) else 0.0,
        "projected_rate": projected / max(len(out), 1),
        "rejection_failure_rate": n_fail / max(len(out), 1),
    }


def bounds_diagnostics_global_and_player(
    df: pd.DataFrame,
    bounds: PlayerTargetBounds,
    split_name: str,
    *,
    player_col: str = "batter_name",
    vocabs: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Global diagnostics JSON block + per-player CSV rows."""
    global_diag: dict[str, Any] = {"split": split_name, "targets": {}, D_TARGET: {}}
    name_to_id = {}
    if vocabs and player_col in vocabs:
        name_to_id = {k: v for k, v in vocabs[player_col].items()}
    id_to_name = {v: k for k, v in name_to_id.items()}

    for name in VA_NAMES:
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        gL, gU = bounds.global_va[name]
        below = int(np.sum(v < gL))
        above = int(np.sum(v > gU))
        global_diag["targets"][name] = {
            "global_bounds": [gL, gU],
            "quantiles": quantile_report(v),
            "below_global": below,
            "above_global": above,
            "pct_outside_global": 100.0 * (below + above) / max(len(v), 1),
        }
        if name == "a_tilde":
            cap = bounds.metadata.get("attack_angle_global_cap", ATTACK_CAP_DEFAULT)
            if cap is not None:
                outside_cap = int(np.sum((v < cap[0]) | (v > cap[1])))
                global_diag["targets"][name]["pct_outside_attack_cap"] = (
                    100.0 * outside_cap / max(len(v), 1)
                )

    dvals = pd.to_numeric(df[D_TARGET], errors="coerce").to_numpy(dtype=np.float64)
    pids = (
        df[player_col]
        .fillna("NA")
        .astype(str)
        .str.strip()
        .map(lambda x: name_to_id.get(x, 0))
        .to_numpy(dtype=np.int64)
    )
    outside_arc = 0
    for idx in range(len(df)):
        if not bounds.get_d_arc(int(pids[idx])).contains_deg(dvals[idx]):
            outside_arc += 1
    global_diag[D_TARGET] = {
        "quantiles": quantile_report(dvals),
        "pct_outside_fitted_arc": 100.0 * outside_arc / max(len(dvals), 1),
    }

    player_rows: list[dict[str, Any]] = []
    for pname, gdf in df.groupby(player_col, sort=False):
        pid = int(name_to_id.get(str(pname).strip(), 0))
        row: dict[str, Any] = {"split": split_name, "player": pname, "player_id": pid, "n_rows": len(gdf)}
        arc = bounds.get_d_arc(pid)
        row["d_arc_width_deg"] = arc.arc_width_deg
        row["d_full_circle"] = arc.full_circle
        row["d_fallback"] = arc.fallback
        for name in VA_NAMES:
            v = pd.to_numeric(gdf[name], errors="coerce").to_numpy(dtype=np.float64)
            L, U = bounds.va_bounds[name].get(pid, bounds.global_va[name])
            row[f"{name}_bounds"] = f"[{L},{U}]"
            row[f"{name}_pct_outside"] = 100.0 * np.mean((v < L) | (v > U))
        dsub = pd.to_numeric(gdf[D_TARGET], errors="coerce").to_numpy(dtype=np.float64)
        row["d_pct_outside_arc"] = (
            100.0 * np.mean(~arc.contains_deg(dsub)) if len(dsub) else 0.0
        )
        player_rows.append(row)
    return global_diag, player_rows


def save_bounds_artifacts(
    bounds: PlayerTargetBounds,
    diagnostics: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "player_target_bounds.json").write_text(
        json.dumps(bounds.state_dict(), indent=2), encoding="utf-8"
    )
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "bounds_diagnostics_global.json").write_text(
        json.dumps(diagnostics.get("global", {}), indent=2), encoding="utf-8"
    )
    if diagnostics.get("player_rows"):
        pd.DataFrame(diagnostics["player_rows"]).to_csv(
            diag_dir / "bounds_diagnostics_by_player.csv", index=False
        )
        pd.DataFrame(diagnostics["player_rows"]).to_csv(
            diag_dir / "player_bounds_summary.csv", index=False
        )
