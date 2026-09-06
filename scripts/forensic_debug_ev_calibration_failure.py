#!/usr/bin/env python3
"""
Forensic EV calibration failure debugger.

Writes a comprehensive plain-text report suitable for external diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def soft_cap(y: np.ndarray, cap: float = 121.0, tau: float = 2.0) -> np.ndarray:
    x = (cap - y) / tau
    return cap - tau * np.logaddexp(0.0, x)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, str):
        return x
    v = _safe_float(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _q(arr: np.ndarray, qv: float) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, qv))


@dataclass
class InferredCols:
    event_id: str
    ev_dec: str
    ev_cal: str | None
    ev_obs: str
    pitch_group: str | None
    split_col: str | None
    draw_id: str | None


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warnings: list[str] = []

    def add(self, s: str = "") -> None:
        self.lines.append(s)

    def warn(self, s: str) -> None:
        self.warnings.append(s)
        self.lines.append(f"WARNING: {s}")

    def section(self, idx: int, title: str) -> None:
        self.add("")
        self.add("=" * 60)
        self.add(f"{idx}. {title}")
        self.add("=" * 60)
        self.add("")

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def attach_obs_to_draws_if_missing(df: pd.DataFrame, draw_path: Path, ev_obs_col_name: str = "observed_EV") -> pd.DataFrame:
    if any(c in df.columns for c in ["EV_obs", "ev_obs", "launch_speed", "observed_EV"]):
        return df
    run_dir = Path(draw_path).resolve().parent
    cand = [run_dir / "predictive_summary_by_event.parquet", run_dir / "predictive_summary_by_event.csv"]
    summ_path = next((p for p in cand if p.exists()), None)
    if summ_path is None:
        return df
    summ = load_table(summ_path)
    eid = "event_id" if "event_id" in summ.columns else ("Event_ID" if "Event_ID" in summ.columns else None)
    if eid is None:
        return df
    obs_c = next((c for c in ["observed_EV", "EV_obs", "launch_speed"] if c in summ.columns), None)
    if obs_c is None:
        return df
    sub = summ[[eid, obs_c]].drop_duplicates(subset=[eid]).copy()
    sub.columns = ["event_id", ev_obs_col_name]
    out = df.copy()
    if "event_id" in out.columns:
        out = out.merge(sub, on="event_id", how="left")
    return out


def infer_columns(df: pd.DataFrame, args: argparse.Namespace, is_cal: bool = False) -> tuple[pd.DataFrame, InferredCols]:
    out = df.copy()
    event_hint = args.event_id_col
    event_candidates = [event_hint] if event_hint else []
    event_candidates += ["event_id", "Event_ID"]
    event_col = None
    for c in event_candidates:
        if c and c in out.columns:
            event_col = c
            break
    if event_col is None:
        if all(c in out.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
            event_col = "__event_id_constructed__"
            out[event_col] = (
                out["game_pk"].astype(str)
                + "_"
                + out["at_bat_number"].astype(str)
                + "_"
                + out["pitch_number"].astype(str)
            )
        else:
            raise ValueError("Could not infer event id column.")

    ev_dec_candidates = []
    if args.ev_dec_col:
        ev_dec_candidates.append(args.ev_dec_col)
    ev_dec_candidates += ["EV_dec", "ev_dec", "EV", "launch_speed_dec", "decoded_EV"]
    ev_dec = next((c for c in ev_dec_candidates if c in out.columns), None)
    if ev_dec is None:
        raise ValueError("Could not infer EV_dec column.")

    ev_obs_candidates = []
    if args.ev_obs_col:
        ev_obs_candidates.append(args.ev_obs_col)
    ev_obs_candidates += ["EV_obs", "ev_obs", "launch_speed", "observed_EV"]
    ev_obs = next((c for c in ev_obs_candidates if c in out.columns), None)
    if ev_obs is None:
        raise ValueError("Could not infer EV_obs column.")

    ev_cal = None
    if is_cal:
        ev_cal_candidates = []
        if args.ev_cal_col:
            ev_cal_candidates.append(args.ev_cal_col)
        ev_cal_candidates += ["EV_cal", "ev_cal", "calibrated_EV"]
        ev_cal = next((c for c in ev_cal_candidates if c in out.columns), None)
        if ev_cal is None:
            raise ValueError("Could not infer EV_cal column from calibrated file.")

    pg_candidates = ([args.pitch_group_col] if args.pitch_group_col else []) + ["pitch_group", "pitch_type"]
    pitch_group = next((c for c in pg_candidates if c and c in out.columns), None)

    split_candidates = ["split", "split_partition", "dataset_split"]
    split_col = next((c for c in split_candidates if c in out.columns), None)

    draw_id_candidates = ["draw_id", "draw_idx", "sample_id", "sim_draw_id"]
    draw_id = next((c for c in draw_id_candidates if c in out.columns), None)

    out[event_col] = out[event_col].astype(str)
    out[ev_dec] = pd.to_numeric(out[ev_dec], errors="coerce")
    out[ev_obs] = pd.to_numeric(out[ev_obs], errors="coerce")
    if ev_cal is not None:
        out[ev_cal] = pd.to_numeric(out[ev_cal], errors="coerce")

    return out, InferredCols(event_col, ev_dec, ev_cal, ev_obs, pitch_group, split_col, draw_id)


def sample_crps(samples: np.ndarray, y: float, max_draws: int, rng: np.random.Generator) -> float:
    s = samples[np.isfinite(samples)]
    if s.size < 2 or not np.isfinite(y):
        return float("nan")
    if s.size > max_draws:
        idx = rng.choice(s.size, size=max_draws, replace=False)
        s = s[idx]
    e1 = np.mean(np.abs(s - y))
    e2 = np.mean(np.abs(s[:, None] - s[None, :]))
    return float(e1 - 0.5 * e2)


def event_table(
    uncal: pd.DataFrame,
    cal: pd.DataFrame,
    cu: InferredCols,
    cc: InferredCols,
) -> pd.DataFrame:
    gb_u = uncal.groupby(cu.event_id, sort=False)
    gb_c = cal.groupby(cc.event_id, sort=False)
    ids = sorted(set(gb_u.groups) & set(gb_c.groups))
    rows: list[dict[str, Any]] = []
    for eid in ids:
        gu = gb_u.get_group(eid)
        gc = gb_c.get_group(eid)
        xu = pd.to_numeric(gu[cu.ev_dec], errors="coerce").to_numpy(dtype=np.float64)
        xc = pd.to_numeric(gc[cc.ev_cal], errors="coerce").to_numpy(dtype=np.float64)
        y = _safe_float(pd.to_numeric(gu[cu.ev_obs], errors="coerce").dropna().iloc[0] if gu[cu.ev_obs].notna().any() else np.nan)
        rows.append(
            {
                "event_id": eid,
                "EV_obs": y,
                "EV_dec_mean": float(np.nanmean(xu)),
                "EV_cal_mean": float(np.nanmean(xc)),
                "EV_dec_std": float(np.nanstd(xu)),
                "EV_cal_std": float(np.nanstd(xc)),
                "EV_dec_p05": _q(xu, 0.05),
                "EV_dec_p50": _q(xu, 0.50),
                "EV_dec_p90": _q(xu, 0.90),
                "EV_dec_p95": _q(xu, 0.95),
                "EV_dec_p99": _q(xu, 0.99),
                "EV_cal_p05": _q(xc, 0.05),
                "EV_cal_p50": _q(xc, 0.50),
                "EV_cal_p90": _q(xc, 0.90),
                "EV_cal_p95": _q(xc, 0.95),
                "EV_cal_p99": _q(xc, 0.99),
                "n_draws_uncal": int(np.isfinite(xu).sum()),
                "n_draws_cal": int(np.isfinite(xc).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["true_residual"] = out["EV_obs"] - out["EV_dec_mean"]
    out["before_error"] = out["EV_dec_mean"] - out["EV_obs"]
    out["after_error"] = out["EV_cal_mean"] - out["EV_obs"]
    out["actual_correction"] = out["EV_cal_mean"] - out["EV_dec_mean"]
    return out


def metrics_block(evt: pd.DataFrame, rep: Reporter) -> dict[str, Any]:
    y = evt["EV_obs"].to_numpy(dtype=np.float64)
    d = evt["EV_dec_mean"].to_numpy(dtype=np.float64)
    c = evt["EV_cal_mean"].to_numpy(dtype=np.float64)
    m: dict[str, Any] = {}
    m["rmse_before"] = float(np.sqrt(mean_squared_error(y, d)))
    m["rmse_after"] = float(np.sqrt(mean_squared_error(y, c)))
    m["mae_before"] = float(mean_absolute_error(y, d))
    m["mae_after"] = float(mean_absolute_error(y, c))
    m["medae_before"] = float(np.median(np.abs(d - y)))
    m["medae_after"] = float(np.median(np.abs(c - y)))
    m["bias_before"] = float(np.mean(d - y))
    m["bias_after"] = float(np.mean(c - y))
    m["median_error_before"] = float(np.median(d - y))
    m["median_error_after"] = float(np.median(c - y))
    m["r2_before"] = float(r2_score(y, d)) if np.var(y) > 1e-12 else float("nan")
    m["r2_after"] = float(r2_score(y, c)) if np.var(y) > 1e-12 else float("nan")
    m["pearson_before"] = float(np.corrcoef(y, d)[0, 1]) if y.size > 2 else float("nan")
    m["pearson_after"] = float(np.corrcoef(y, c)[0, 1]) if y.size > 2 else float("nan")
    m["spearman_before"] = float(pd.Series(y).corr(pd.Series(d), method="spearman"))
    m["spearman_after"] = float(pd.Series(y).corr(pd.Series(c), method="spearman"))
    rep.add(f"RMSE: {_fmt(m['rmse_before'])} -> {_fmt(m['rmse_after'])}")
    rep.add(f"MAE: {_fmt(m['mae_before'])} -> {_fmt(m['mae_after'])}")
    rep.add(f"Bias: {_fmt(m['bias_before'])} -> {_fmt(m['bias_after'])}")
    rep.add(f"CRPS / coverage / PIT are reported below.")
    return m


def interval_pit_crps(
    uncal: pd.DataFrame,
    cal: pd.DataFrame,
    cu: InferredCols,
    cc: InferredCols,
    levels: list[float],
    max_events_for_crps: int,
    max_draws_per_event_crps: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    gb_u = uncal.groupby(cu.event_id, sort=False)
    gb_c = cal.groupby(cc.event_id, sort=False)
    ids = sorted(set(gb_u.groups) & set(gb_c.groups))
    rows: list[dict[str, Any]] = []
    pit_b: list[float] = []
    pit_a: list[float] = []
    crps_b: list[float] = []
    crps_a: list[float] = []
    ids_for_crps = ids[:max_events_for_crps]
    for eid in ids:
        gu = gb_u.get_group(eid)
        gc = gb_c.get_group(eid)
        xu = pd.to_numeric(gu[cu.ev_dec], errors="coerce").to_numpy(dtype=np.float64)
        xc = pd.to_numeric(gc[cc.ev_cal], errors="coerce").to_numpy(dtype=np.float64)
        yo = pd.to_numeric(gu[cu.ev_obs], errors="coerce").dropna()
        y = float(yo.iloc[0]) if len(yo) else float("nan")
        if np.isfinite(y):
            u_fin = xu[np.isfinite(xu)]
            c_fin = xc[np.isfinite(xc)]
            if u_fin.size:
                pit_b.append(float(np.mean(u_fin <= y)))
            if c_fin.size:
                pit_a.append(float(np.mean(c_fin <= y)))
            if eid in ids_for_crps:
                crps_b.append(sample_crps(u_fin, y, max_draws_per_event_crps, rng))
                crps_a.append(sample_crps(c_fin, y, max_draws_per_event_crps, rng))
            for a in levels:
                ql = (1.0 - a) / 2.0
                qh = 1.0 - ql
                lo_u, hi_u = np.quantile(u_fin, [ql, qh])
                lo_c, hi_c = np.quantile(c_fin, [ql, qh])
                rows.append(
                    {
                        "event_id": eid,
                        "alpha": a,
                        "cover_before": float(lo_u <= y <= hi_u),
                        "cover_after": float(lo_c <= y <= hi_c),
                        "width_before": float(hi_u - lo_u),
                        "width_after": float(hi_c - lo_c),
                    }
                )
    iv = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    out["mean_crps_before"] = float(np.nanmean(crps_b)) if crps_b else float("nan")
    out["mean_crps_after"] = float(np.nanmean(crps_a)) if crps_a else float("nan")
    out["crps_percent_improvement"] = (
        float(100.0 * (out["mean_crps_before"] - out["mean_crps_after"]) / out["mean_crps_before"])
        if np.isfinite(out["mean_crps_before"]) and out["mean_crps_before"] > 0
        else float("nan")
    )
    out["pit_mean_before"] = float(np.nanmean(pit_b)) if pit_b else float("nan")
    out["pit_std_before"] = float(np.nanstd(pit_b)) if pit_b else float("nan")
    out["pit_mean_after"] = float(np.nanmean(pit_a)) if pit_a else float("nan")
    out["pit_std_after"] = float(np.nanstd(pit_a)) if pit_a else float("nan")
    out["pit_below_0_05_before"] = float(np.mean(np.array(pit_b) < 0.05)) if pit_b else float("nan")
    out["pit_above_0_95_before"] = float(np.mean(np.array(pit_b) > 0.95)) if pit_b else float("nan")
    out["pit_below_0_05_after"] = float(np.mean(np.array(pit_a) < 0.05)) if pit_a else float("nan")
    out["pit_above_0_95_after"] = float(np.mean(np.array(pit_a) > 0.95)) if pit_a else float("nan")
    out["pit_hist10_before"] = np.histogram(pit_b, bins=10, range=(0, 1))[0].tolist() if pit_b else []
    out["pit_hist10_after"] = np.histogram(pit_a, bins=10, range=(0, 1))[0].tolist() if pit_a else []
    cov_rows: list[dict[str, Any]] = []
    if not iv.empty:
        for a, g in iv.groupby("alpha", sort=False):
            eb = float(g["cover_before"].mean())
            ea = float(g["cover_after"].mean())
            cov_rows.append(
                {
                    "alpha": float(a),
                    "empirical_coverage_before": eb,
                    "empirical_coverage_after": ea,
                    "coverage_error_before": eb - float(a),
                    "coverage_error_after": ea - float(a),
                    "mean_width_before": float(g["width_before"].mean()),
                    "mean_width_after": float(g["width_after"].mean()),
                    "median_width_before": float(g["width_before"].median()),
                    "median_width_after": float(g["width_after"].median()),
                }
            )
    return out, pd.DataFrame(cov_rows)


def tail_block(
    uncal: pd.DataFrame,
    cal: pd.DataFrame,
    cu: InferredCols,
    cc: InferredCols,
    thresholds: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    xb = pd.to_numeric(uncal[cu.ev_dec], errors="coerce").to_numpy(dtype=np.float64)
    xa = pd.to_numeric(cal[cc.ev_cal], errors="coerce").to_numpy(dtype=np.float64)
    xb = xb[np.isfinite(xb)]
    xa = xa[np.isfinite(xa)]
    out: dict[str, Any] = {
        "max_before": float(np.max(xb)),
        "max_after": float(np.max(xa)),
        "p95_before": _q(xb, 0.95),
        "p95_after": _q(xa, 0.95),
        "p99_before": _q(xb, 0.99),
        "p99_after": _q(xa, 0.99),
        "p995_before": _q(xb, 0.995),
        "p995_after": _q(xa, 0.995),
        "p999_before": _q(xb, 0.999),
        "p999_after": _q(xa, 0.999),
    }
    y_evt = pd.to_numeric(uncal.groupby(cu.event_id, sort=False)[cu.ev_obs].first(), errors="coerce").dropna().to_numpy()
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        pb = float(np.mean(xb > t))
        pa = float(np.mean(xa > t))
        po = float(np.mean(y_evt > t))
        out[f"frac_gt_{t}_before"] = pb
        out[f"frac_gt_{t}_after"] = pa
        rows.append(
            {
                "threshold": float(t),
                "observed_exceedance_rate": po,
                "predicted_before": pb,
                "predicted_after": pa,
                "abs_error_before": abs(pb - po),
                "abs_error_after": abs(pa - po),
            }
        )
    return out, pd.DataFrame(rows)


def split_shift_block(
    evt_test: pd.DataFrame,
    train_draws: pd.DataFrame | None,
    cu: InferredCols,
    rep: Reporter,
) -> pd.DataFrame | None:
    if train_draws is None:
        rep.add("No calibration train/cal draws supplied; split-shift audit skipped.")
        return None
    gb = train_draws.groupby(cu.event_id, sort=False)
    rows = []
    for eid, g in gb:
        x = pd.to_numeric(g[cu.ev_dec], errors="coerce").to_numpy(dtype=np.float64)
        yv = pd.to_numeric(g[cu.ev_obs], errors="coerce").dropna()
        y = float(yv.iloc[0]) if len(yv) else float("nan")
        rows.append(
            {
                "event_id": str(eid),
                "EV_dec_mean": float(np.nanmean(x)),
                "EV_dec_std": float(np.nanstd(x)),
                "EV_dec_p90": _q(x, 0.90),
                "EV_dec_p95": _q(x, 0.95),
                "EV_dec_p99": _q(x, 0.99),
                "EV_obs": y,
                "true_residual": y - float(np.nanmean(x)),
                "split_source": "train_cal",
            }
        )
    tr = pd.DataFrame(rows)
    te = evt_test.copy()
    te["split_source"] = "test"
    cols = ["EV_dec_mean", "EV_dec_std", "EV_dec_p90", "EV_dec_p95", "EV_dec_p99", "EV_obs", "true_residual"]
    rep.add("Train/cal vs test summary (means):")
    for c in cols:
        rep.add(f"- {c}: train_cal={_fmt(tr[c].mean())}, test={_fmt(te[c].mean())}")
    if np.isfinite(tr["true_residual"].mean()) and np.isfinite(te["true_residual"].mean()):
        if np.sign(tr["true_residual"].mean()) != np.sign(te["true_residual"].mean()):
            rep.warn("Train/cal residual mean sign differs from test residual mean sign.")
    return tr


def baseline_comparison(
    uncal: pd.DataFrame,
    cal: pd.DataFrame,
    evt: pd.DataFrame,
    cu: InferredCols,
    cc: InferredCols,
    thresholds: list[float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    gb_u = uncal.groupby(cu.event_id, sort=False)
    gb_c = cal.groupby(cc.event_id, sort=False)
    ids = sorted(set(gb_u.groups) & set(gb_c.groups))
    rows = []
    y_evt_map = evt.set_index("event_id")["EV_obs"].to_dict()
    global_shift = float(np.mean(evt["EV_obs"] - evt["EV_dec_mean"]))
    lr = LinearRegression().fit(evt[["EV_dec_mean"]], evt["EV_obs"])
    for name in ["no_calibration", "softcap_only", "global_shift", "global_affine", "current_learned"]:
        ev_mean_pred = []
        y_obs = []
        draw_vals: list[float] = []
        crps_vals = []
        cov90 = []
        wid90 = []
        for eid in ids:
            gu = gb_u.get_group(eid)
            gc = gb_c.get_group(eid)
            xu = pd.to_numeric(gu[cu.ev_dec], errors="coerce").to_numpy(dtype=np.float64)
            xc = pd.to_numeric(gc[cc.ev_cal], errors="coerce").to_numpy(dtype=np.float64)
            y = float(y_evt_map.get(eid, np.nan))
            if not np.isfinite(y):
                continue
            if name == "no_calibration":
                x = xu
            elif name == "softcap_only":
                x = soft_cap(xu, cap=121.0, tau=2.0)
            elif name == "global_shift":
                x = xu + global_shift
            elif name == "global_affine":
                x = lr.intercept_ + lr.coef_[0] * xu
            else:
                x = xc
            x = x[np.isfinite(x)]
            if x.size < 2:
                continue
            ev_mean_pred.append(float(np.mean(x)))
            y_obs.append(y)
            draw_vals.extend(x.tolist())
            crps_vals.append(sample_crps(x, y, 2000, rng))
            ql, qh = np.quantile(x, [0.05, 0.95])
            cov90.append(float(ql <= y <= qh))
            wid90.append(float(qh - ql))
        if not y_obs:
            continue
        y_arr = np.array(y_obs)
        p_arr = np.array(ev_mean_pred)
        draw_arr = np.array(draw_vals)
        row = {
            "baseline": name,
            "rmse": float(np.sqrt(mean_squared_error(y_arr, p_arr))),
            "mae": float(mean_absolute_error(y_arr, p_arr)),
            "bias": float(np.mean(p_arr - y_arr)),
            "crps": float(np.nanmean(crps_vals)),
            "cov90": float(np.mean(cov90)),
            "width90": float(np.mean(wid90)),
            "max_draw": float(np.max(draw_arr)),
            "p99_draw": _q(draw_arr, 0.99),
        }
        for t in thresholds:
            row[f"p_gt_{t}"] = float(np.mean(draw_arr > t))
        rows.append(row)
    return pd.DataFrame(rows)


def inspect_calibrator(path: Path, rep: Reporter, evt: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        rep.add("No calibrator artifact provided/found.")
        return out
    obj = joblib.load(path)
    rep.add(f"Loaded calibrator object type: {type(obj)}")
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    rep.add(f"Public attrs (first 60): {attrs[:60]}")
    for key in ("rho", "cap", "tau", "ev_col", "group_col", "numeric_features", "categorical_features"):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
            rep.add(f"{key}: {getattr(obj, key)}")
    if hasattr(obj, "pipeline_"):
        rep.add(f"pipeline_: {obj.pipeline_}")
    if hasattr(obj, "predict_residual"):
        feats = []
        for c in (getattr(obj, "numeric_features", []) + getattr(obj, "categorical_features", [])):
            if c in evt.columns:
                feats.append(c)
        if feats:
            dfm = evt.copy()
            for c in feats:
                if c not in dfm.columns:
                    dfm[c] = np.nan
            try:
                pr = np.asarray(obj.predict_residual(dfm), dtype=float)
                out["predicted_residual_summary"] = {
                    "mean": float(np.nanmean(pr)),
                    "std": float(np.nanstd(pr)),
                    "p05": _q(pr, 0.05),
                    "p95": _q(pr, 0.95),
                }
            except Exception as e:
                rep.warn(f"predict_residual failed during forensic check: {e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncalibrated-draws-path", type=Path, required=True)
    ap.add_argument("--calibrated-draws-path", type=Path, required=True)
    ap.add_argument("--calibrator-path", type=Path, default=None)
    ap.add_argument("--calibration-train-draws-path", type=Path, default=None)
    ap.add_argument("--calibration-summary-path", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--event-id-col", type=str, default=None)
    ap.add_argument("--ev-dec-col", type=str, default=None)
    ap.add_argument("--ev-cal-col", type=str, default=None)
    ap.add_argument("--ev-obs-col", type=str, default=None)
    ap.add_argument("--pitch-group-col", type=str, default=None)
    ap.add_argument("--max-events-for-crps", type=int, default=5000)
    ap.add_argument("--max-draws-per-event-crps", type=int, default=2000)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = args.output_dir or (Path("artifacts/physics_decoder_calibration_forensics") / stamp)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ev_calibration_failure_report.txt"
    rng = np.random.default_rng(args.random_seed)
    rep = Reporter()

    rep.section(0, "EXECUTIVE SUMMARY")
    rep.add("This report audits EV-only calibration failure with recomputed metrics, alignment checks,")
    rep.add("residual-sign algebra checks, tail and coverage diagnostics, shift analysis, and baseline tests.")
    rep.add("Main suspected failure modes are ranked near the end after evidence is computed.")

    rep.section(1, "FILES, SHAPES, AND BASIC AUDIT")
    paths = {
        "uncalibrated_draws": args.uncalibrated_draws_path,
        "calibrated_draws": args.calibrated_draws_path,
        "calibrator": args.calibrator_path,
        "calibration_train_draws": args.calibration_train_draws_path,
        "calibration_summary": args.calibration_summary_path,
    }
    for k, p in paths.items():
        if p is None:
            rep.add(f"{k}: None")
        else:
            rep.add(f"{k}: {p} exists={p.exists()} size={p.stat().st_size if p.exists() else 'NA'}")

    uncal_raw = load_table(args.uncalibrated_draws_path)
    cal_raw = load_table(args.calibrated_draws_path)
    uncal, cu = infer_columns(uncal_raw, args, is_cal=False)
    cal, cc = infer_columns(cal_raw, args, is_cal=True)
    rep.add(f"Inferred columns (uncal): event={cu.event_id}, EV_dec={cu.ev_dec}, EV_obs={cu.ev_obs}, pitch_group={cu.pitch_group}, split={cu.split_col}, draw_id={cu.draw_id}")
    rep.add(f"Inferred columns (cal): event={cc.event_id}, EV_cal={cc.ev_cal}, EV_obs={cc.ev_obs}, pitch_group={cc.pitch_group}, split={cc.split_col}, draw_id={cc.draw_id}")
    rep.add(f"uncal shape={uncal.shape}, cal shape={cal.shape}")
    rep.add(f"uncal unique events={uncal[cu.event_id].nunique()}, cal unique events={cal[cc.event_id].nunique()}")
    rep.add(f"uncal draws/event mean={_fmt(uncal.groupby(cu.event_id).size().mean())}, median={_fmt(uncal.groupby(cu.event_id).size().median())}, max={_fmt(uncal.groupby(cu.event_id).size().max())}")
    rep.add(f"cal draws/event mean={_fmt(cal.groupby(cc.event_id).size().mean())}, median={_fmt(cal.groupby(cc.event_id).size().median())}, max={_fmt(cal.groupby(cc.event_id).size().max())}")
    rep.add(f"uncal duplicate rows={int(uncal.duplicated().sum())}, cal duplicate rows={int(cal.duplicated().sum())}")
    rep.add(f"uncal missing key cols: EV_dec={int(uncal[cu.ev_dec].isna().sum())}, EV_obs={int(uncal[cu.ev_obs].isna().sum())}, event_id={int(uncal[cu.event_id].isna().sum())}")
    rep.add(f"cal missing key cols: EV_cal={int(cal[cc.ev_cal].isna().sum())}, EV_obs={int(cal[cc.ev_obs].isna().sum())}, event_id={int(cal[cc.event_id].isna().sum())}")
    rep.add(f"memory usage bytes: uncal={int(uncal.memory_usage(deep=True).sum())}, cal={int(cal.memory_usage(deep=True).sum())}")

    if np.allclose(
        uncal[cu.ev_dec].fillna(0).to_numpy(dtype=float),
        uncal[cu.ev_obs].fillna(0).to_numpy(dtype=float),
        atol=1e-10,
    ):
        rep.warn("EV_dec appears identical to EV_obs in uncalibrated file.")
    if cc.ev_cal and cc.ev_cal in cal.columns:
        if np.allclose(
            cal[cc.ev_cal].fillna(0).to_numpy(dtype=float),
            cal[cc.ev_obs].fillna(0).to_numpy(dtype=float),
            atol=1e-10,
        ):
            rep.warn("EV_cal appears identical to EV_obs in calibrated file.")

    rep.section(2, "EVENT MATCHING AUDIT")
    ids_u = set(uncal[cu.event_id].astype(str).unique())
    ids_c = set(cal[cc.event_id].astype(str).unique())
    rep.add(f"events in uncalibrated={len(ids_u)}")
    rep.add(f"events in calibrated={len(ids_c)}")
    rep.add(f"common events={len(ids_u & ids_c)}")
    rep.add(f"only in uncalibrated={len(ids_u - ids_c)}")
    rep.add(f"only in calibrated={len(ids_c - ids_u)}")
    if ids_u != ids_c:
        rep.warn("Event sets differ between uncalibrated and calibrated files.")
    dc_u = uncal.groupby(cu.event_id).size().rename("n_u")
    dc_c = cal.groupby(cc.event_id).size().rename("n_c")
    dc = dc_u.to_frame().join(dc_c, how="inner")
    diff = (dc["n_u"] - dc["n_c"]).abs()
    rep.add(f"draw count mean diff per event={_fmt(diff.mean())}, max abs diff={_fmt(diff.max())}, mismatched events={int((diff>0).sum())}")
    if int((diff > 0).sum()) > 0:
        rep.warn("Per-event draw counts mismatch between uncalibrated and calibrated files.")
    if cu.draw_id and cc.draw_id and cu.draw_id in uncal.columns and cc.draw_id in cal.columns:
        rep.add("Draw-level IDs exist; alignment can be further checked externally.")
    else:
        rep.add("No reliable draw-level ID in both files; only event-level alignment verified.")
    y_u = pd.to_numeric(uncal.groupby(cu.event_id)[cu.ev_obs].first(), errors="coerce")
    y_c = pd.to_numeric(cal.groupby(cc.event_id)[cc.ev_obs].first(), errors="coerce")
    yj = y_u.to_frame("yu").join(y_c.to_frame("yc"), how="inner")
    ydiff = (yj["yu"] - yj["yc"]).abs()
    rep.add(f"EV_obs max abs difference per event={_fmt(ydiff.max())}, differing events={int((ydiff>1e-9).sum())}")
    if int((ydiff > 1e-9).sum()) > 0:
        rep.warn("EV_obs differs across calibrated vs uncalibrated file for same event.")
        rep.add("Sample mismatches:")
        rep.add(str(yj.loc[ydiff > 1e-9].head(10)))

    evt = event_table(uncal, cal, cu, cc)
    evt.to_csv(out_dir / "event_level_forensics.csv", index=False)

    rep.section(3, "RECOMPUTED BEFORE/AFTER METRICS")
    mean_metrics = metrics_block(evt, rep)
    iv_metrics, iv_df = interval_pit_crps(
        uncal,
        cal,
        cu,
        cc,
        levels=[0.50, 0.80, 0.90, 0.95],
        max_events_for_crps=args.max_events_for_crps,
        max_draws_per_event_crps=args.max_draws_per_event_crps,
        rng=rng,
    )
    iv_df.to_csv(out_dir / "interval_coverage_table.csv", index=False)
    tail_metrics, tail_df = tail_block(uncal, cal, cu, cc, thresholds=[100, 105, 110, 115, 120])
    tail_df.to_csv(out_dir / "tail_exceedance_comparison.csv", index=False)
    rep.add(f"CRPS mean: {_fmt(iv_metrics['mean_crps_before'])} -> {_fmt(iv_metrics['mean_crps_after'])}")
    rep.add(f"CRPS improvement (%): {_fmt(iv_metrics['crps_percent_improvement'])}")
    rep.add(f"PIT mean/std before=({_fmt(iv_metrics['pit_mean_before'])}, {_fmt(iv_metrics['pit_std_before'])})")
    rep.add(f"PIT mean/std after =({_fmt(iv_metrics['pit_mean_after'])}, {_fmt(iv_metrics['pit_std_after'])})")
    rep.add("Coverage table:")
    rep.add(iv_df.to_string(index=False))
    rep.add("Tail metrics:")
    rep.add(json.dumps(tail_metrics, indent=2))

    rep.section(4, "RESIDUAL SIGN AND ALGEBRA AUDIT")
    tr = evt["true_residual"].to_numpy(dtype=float)
    ac = evt["actual_correction"].to_numpy(dtype=float)
    be = evt["before_error"].to_numpy(dtype=float)
    ae = evt["after_error"].to_numpy(dtype=float)
    rep.add(f"mean(true_residual)={_fmt(np.mean(tr))}")
    rep.add(f"median(true_residual)={_fmt(np.median(tr))}")
    rep.add(f"mean(before_error)={_fmt(np.mean(be))}")
    rep.add(f"mean(after_error)={_fmt(np.mean(ae))}")
    rep.add(f"mean(actual_correction)={_fmt(np.mean(ac))}")
    rep.add(f"median(actual_correction)={_fmt(np.median(ac))}")
    rep.add(f"std(actual_correction)={_fmt(np.std(ac))}")
    rep.add(
        "actual_correction quantiles p01/p05/p25/p50/p75/p95/p99="
        + ", ".join(_fmt(_q(ac, q)) for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    )
    rep.add(f"share(actual_correction>0)={_fmt(np.mean(ac>0))}")
    rep.add(f"share(actual_correction<0)={_fmt(np.mean(ac<0))}")
    rep.add(f"share(true_residual>0)={_fmt(np.mean(tr>0))}")
    rep.add(f"share(true_residual<0)={_fmt(np.mean(tr<0))}")
    if np.mean(be) > 0 and np.mean(ac) > 0:
        rep.warn("Decoder overpredicts on average, but actual correction is positive on average.")
    if abs(np.mean(ae)) > abs(np.mean(be)):
        rep.warn("Calibration moved average bias farther from zero.")
    sign_agree = np.mean(np.sign(tr) == np.sign(ac))
    rep.add(f"sign agreement rate true_residual vs actual_correction={_fmt(sign_agree)}")
    if sign_agree < 0.50:
        rep.warn("Sign agreement between true residual and correction is below 50%.")
    corr = np.corrcoef(tr, ac)[0, 1] if len(tr) > 2 else float("nan")
    rep.add(f"corr(true_residual, actual_correction)={_fmt(corr)}")
    rep.add(f"RMSE(actual_correction,true_residual)={_fmt(np.sqrt(np.mean((ac-tr)**2)))}")
    rep.add(f"MAE(actual_correction,true_residual)={_fmt(np.mean(np.abs(ac-tr)))}")
    rep.add(f"mean(actual_correction-true_residual)={_fmt(np.mean(ac-tr))}")
    rep.add(f"median(actual_correction-true_residual)={_fmt(np.median(ac-tr))}")

    calib_info = {}
    if args.calibrator_path:
        calib_info = inspect_calibrator(args.calibrator_path, rep, evt)

    rep.section(5, "SOFT CAP AND SHRINKAGE AUDIT")
    cap = 121.0
    xb = pd.to_numeric(uncal[cu.ev_dec], errors="coerce").to_numpy(dtype=float)
    xa = pd.to_numeric(cal[cc.ev_cal], errors="coerce").to_numpy(dtype=float)
    xb = xb[np.isfinite(xb)]
    xa = xa[np.isfinite(xa)]
    rep.add(f"uncal draws above cap({cap}) count={int(np.sum(xb>cap))}, frac={_fmt(np.mean(xb>cap))}")
    rep.add(f"cal draws near cap(>=120.99) count={int(np.sum(xa>=120.99))}, frac={_fmt(np.mean(xa>=120.99))}")
    delta = xa - xb[: len(xa)] if len(xa) == len(xb) else np.array([])
    if delta.size:
        for t in [0.1, 1, 3, 5, 10]:
            rep.add(f"share |delta|>{t}: {_fmt(np.mean(np.abs(delta)>t))}")
    rep.add(f"draw mean before/after: {_fmt(np.mean(xb))} -> {_fmt(np.mean(xa))}")
    rep.add(f"draw median before/after: {_fmt(np.median(xb))} -> {_fmt(np.median(xa))}")
    rep.add(f"draw p95 before/after: {_fmt(_q(xb,0.95))} -> {_fmt(_q(xa,0.95))}")
    rep.add(f"draw p99 before/after: {_fmt(_q(xb,0.99))} -> {_fmt(_q(xa,0.99))}")
    std_ratio = (evt["EV_cal_std"] / evt["EV_dec_std"].replace(0, np.nan)).to_numpy(dtype=float)
    rep.add(
        "EV_cal_std / EV_dec_std mean/median/p05/p95="
        + ", ".join(_fmt(v) for v in [np.nanmean(std_ratio), np.nanmedian(std_ratio), _q(std_ratio, 0.05), _q(std_ratio, 0.95)])
    )

    rep.section(6, "TRAIN/CAL/TEST DISTRIBUTION SHIFT AUDIT")
    train_draws = None
    if args.calibration_train_draws_path and Path(args.calibration_train_draws_path).exists():
        tr_raw = load_table(args.calibration_train_draws_path)
        tr_raw = attach_obs_to_draws_if_missing(tr_raw, Path(args.calibration_train_draws_path))
        train_draws, _ = infer_columns(tr_raw, args, is_cal=False)
    tr_evt = split_shift_block(evt, train_draws, cu, rep)

    rep.section(7, "PITCH GROUP / CONTEXT BREAKDOWN")
    pg = cu.pitch_group
    if pg and pg in uncal.columns:
        gp = uncal.groupby(cu.event_id, sort=False)[pg].first().rename("pitch_group").reset_index()
        evg = evt.merge(gp, on="event_id", how="left")
        rows = []
        for g, s in evg.groupby("pitch_group", dropna=False):
            ids = set(s["event_id"])
            xu = uncal[uncal[cu.event_id].isin(ids)][cu.ev_dec].to_numpy(dtype=float)
            xc = cal[cal[cc.event_id].isin(ids)][cc.ev_cal].to_numpy(dtype=float)
            y = s["EV_obs"].to_numpy(dtype=float)
            d = s["EV_dec_mean"].to_numpy(dtype=float)
            c = s["EV_cal_mean"].to_numpy(dtype=float)
            rows.append(
                {
                    "pitch_group": str(g),
                    "n_events": int(len(s)),
                    "n_draws": int(np.isfinite(xu).sum()),
                    "obs_mean": float(np.mean(y)),
                    "dec_mean_avg": float(np.mean(d)),
                    "cal_mean_avg": float(np.mean(c)),
                    "rmse_before": float(np.sqrt(mean_squared_error(y, d))),
                    "rmse_after": float(np.sqrt(mean_squared_error(y, c))),
                    "bias_before": float(np.mean(d - y)),
                    "bias_after": float(np.mean(c - y)),
                    "p99_draw_before": _q(xu, 0.99),
                    "p99_draw_after": _q(xc, 0.99),
                    "p_draw_gt_110_before": float(np.mean(xu > 110)),
                    "p_draw_gt_110_after": float(np.mean(xc > 110)),
                    "p_draw_gt_115_before": float(np.mean(xu > 115)),
                    "p_draw_gt_115_after": float(np.mean(xc > 115)),
                    "obs_p_gt_115": float(np.mean(y > 115)),
                    "mean_actual_correction": float(np.mean(s["actual_correction"])),
                }
            )
        pg_df = pd.DataFrame(rows)
        pg_df.to_csv(out_dir / "pitch_group_forensics.csv", index=False)
        rep.add("Pitch-group diagnostics:")
        rep.add(pg_df.to_string(index=False))
        rep.add("Top RMSE deterioration groups:")
        rep.add(pg_df.assign(rmse_deterioration=pg_df["rmse_after"] - pg_df["rmse_before"]).sort_values("rmse_deterioration", ascending=False).head(10).to_string(index=False))
    else:
        rep.add("No pitch_group/pitch_type column found; section skipped.")

    rep.section(8, "SIMPLE BASELINE COMPARISON")
    base_df = baseline_comparison(uncal, cal, evt, cu, cc, [110, 115, 120], rng)
    base_df.to_csv(out_dir / "baseline_comparison.csv", index=False)
    rep.add(base_df.sort_values("crps").to_string(index=False))
    best = base_df.sort_values("crps").iloc[0]["baseline"] if not base_df.empty else "none"
    rep.add(f"Best baseline by CRPS: {best}")
    if not base_df.empty and "current_learned" in set(base_df["baseline"]):
        cur = base_df.loc[base_df["baseline"] == "current_learned"].iloc[0]
        better = base_df[base_df["crps"] < cur["crps"]]
        if len(better) > 0:
            rep.warn(f"Dumb/alternative baselines beat learned calibrator by CRPS: {better['baseline'].tolist()}")

    rep.section(9, "CALIBRATOR ARTIFACT INSPECTION")
    if args.calibrator_path:
        rep.add(json.dumps(calib_info, indent=2, default=str))
    else:
        rep.add("No calibrator path provided.")

    rep.section(10, "LEAKAGE AND TARGET AUDIT")
    suspicious_tokens = ["obs", "observed", "launch_speed", "target", "residual", "EV_cal", "woba", "events", "hit"]
    sus_cols = []
    for c in set(list(uncal.columns) + list(cal.columns)):
        lc = c.lower()
        if any(t.lower() in lc for t in suspicious_tokens):
            sus_cols.append(c)
    rep.add(f"Suspicious columns found ({len(sus_cols)}): {sorted(sus_cols)[:200]}")
    y = pd.to_numeric(uncal[cu.ev_obs], errors="coerce")
    rep.add(f"EV_obs summary min/max/mean/std={_fmt(y.min())}, {_fmt(y.max())}, {_fmt(y.mean())}, {_fmt(y.std())}")
    rep.add(f"EV_obs missing proportion={_fmt(y.isna().mean())}")
    rep.add(f"EV_obs implausible (<20 or >130) proportion={_fmt(((y<20)|(y>130)).mean())}")

    rep.section(11, "ROW-LEVEL WORST CASES")
    evt["error_deterioration"] = np.abs(evt["after_error"]) - np.abs(evt["before_error"])
    worst = evt.sort_values("error_deterioration", ascending=False).head(20).copy()
    cond = evt[(evt["EV_dec_mean"] > evt["EV_obs"]) & (evt["EV_cal_mean"] > evt["EV_dec_mean"])].head(20).copy()
    opp = evt[np.sign(evt["true_residual"]) != np.sign(evt["actual_correction"])].head(20).copy()
    worst.to_csv(out_dir / "worst_case_events.csv", index=False)
    rep.add("Top 20 events by deterioration:")
    rep.add(worst.to_string(index=False))
    rep.add("Top 20 shifted upward despite overprediction:")
    rep.add(cond.to_string(index=False))
    rep.add("Top 20 opposite-sign residual/correction:")
    rep.add(opp.to_string(index=False))

    rep.section(12, "FINAL DIAGNOSIS")
    # heuristic diagnosis
    rmse_worse = _safe_float(mean_metrics["rmse_after"]) > _safe_float(mean_metrics["rmse_before"])
    crps_worse = _safe_float(iv_metrics["mean_crps_after"]) > _safe_float(iv_metrics["mean_crps_before"])
    bias_worse = abs(_safe_float(mean_metrics["bias_after"])) > abs(_safe_float(mean_metrics["bias_before"]))
    cov90_row = iv_df[iv_df["alpha"].eq(0.90)] if not iv_df.empty else pd.DataFrame()
    cov_worse = False
    if not cov90_row.empty:
        eb = float(cov90_row["empirical_coverage_before"].iloc[0])
        ea = float(cov90_row["empirical_coverage_after"].iloc[0])
        cov_worse = abs(ea - 0.90) > abs(eb - 0.90) + 1e-12
    tail_impossible_reduced = _safe_float(tail_metrics["max_after"]) < _safe_float(tail_metrics["max_before"])
    rep.add(f"A. Wrong residual direction? {'yes' if np.mean(be) > 0 and np.mean(ac) > 0 else 'unclear'}")
    rep.add(f"B. Learned bad residual corrections? {'yes' if rmse_worse and bias_worse else 'unclear'}")
    rep.add(f"C. Variance too aggressively reduced? {'yes' if cov_worse and (_safe_float(_q(xa,0.99)) < _safe_float(_q(xb,0.99))) else 'unclear'}")
    rep.add(f"D. Train/test distribution shift? {'yes' if tr_evt is not None and abs(tr_evt['true_residual'].mean()-evt['true_residual'].mean())>0.5 else 'unclear'}")
    rep.add(f"E. Fixed max EV but not distribution? {'yes' if tail_impossible_reduced and (rmse_worse or crps_worse) else 'unclear'}")
    rep.add(f"F. Dumb baselines beat learned calibrator? {'yes' if ('current_learned' in set(base_df['baseline']) and any(base_df['crps'] < base_df.loc[base_df['baseline'].eq('current_learned'),'crps'].iloc[0])) else 'unclear'}")
    rep.add("")
    rep.add("Most likely root cause:")
    root = []
    if rmse_worse and crps_worse:
        root.append("calibration shifts the event mean in the wrong direction for many events")
    if bias_worse:
        root.append("average correction pushes predictions farther from observed EV")
    if tail_impossible_reduced:
        root.append("soft-cap/shrinkage reduces extreme tails but does not preserve central calibration")
    if not root:
        root.append("mixed effects; no single dominant failure mode")
    rep.add("- " + "; ".join(root) + ".")
    rep.add("")
    rep.add("Recommended next experiment:")
    rep.add(
        "- Fit a simpler baseline (global shift + optional affine + mild shrinkage), "
        "ablate soft-cap and rho separately, and retune with objective combining RMSE+CRPS+coverage "
        "on validation before touching test."
    )

    rep.save(report_path)
    print(str(report_path))


if __name__ == "__main__":
    main()
