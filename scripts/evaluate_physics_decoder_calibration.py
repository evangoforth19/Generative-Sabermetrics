#!/usr/bin/env python3
"""
Held-out diagnostics for EV-only physics decoder calibration (before vs after).

Compares uncalibrated decoder EV draws to calibrated draws on the same events.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parents[1]
MMC2 = REPO / "MCMC 2"
if str(MMC2) not in sys.path:
    sys.path.insert(0, str(MMC2))


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def load_draws(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def infer_columns(
    df_in: pd.DataFrame,
    *,
    event_id_col_hint: str | None,
    ev_dec_candidates: list[str],
    ev_cal_candidates: list[str],
    ev_obs_candidates: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    df = df_in.copy()
    evt = event_id_col_hint if event_id_col_hint and event_id_col_hint in df.columns else "event_id"
    if evt not in df.columns:
        if all(c in df.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
            evt = "__event_comp__"
            df[evt] = (
                df["game_pk"].astype(str)
                + "_"
                + df["at_bat_number"].astype(str)
                + "_"
                + df["pitch_number"].astype(str)
            )
        else:
            evt = event_id_col_hint or "event_id"

    def pick(candidates: list[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"Could not infer column among {candidates}. Available sample: {list(df.columns)[:40]}"
        )

    ev_dec = pick(ev_dec_candidates)
    ev_obs = pick(ev_obs_candidates)
    ev_cal = None
    for c in ev_cal_candidates:
        if c in df.columns:
            ev_cal = c
            break
    return df, {"event_col": evt, "ev_dec": ev_dec, "ev_cal": ev_cal, "ev_obs": ev_obs}


def build_event_level_table(
    df_uncal: pd.DataFrame,
    df_cal: pd.DataFrame | None,
    cols_uncal: dict[str, str],
    cols_cal: dict[str, str] | None,
) -> pd.DataFrame:
    gc = cols_uncal["event_col"]
    ed = cols_uncal["ev_dec"]
    eo = cols_uncal["ev_obs"]

    du = df_uncal.copy()
    du["_ev_dec"] = pd.to_numeric(du[ed], errors="coerce")
    grp_u = du.groupby(gc, sort=False)["_ev_dec"].agg(["mean", "count"]).rename(
        columns={"mean": "EV_dec_mean", "count": "n_draws_dec"}
    )
    eo_u = du.groupby(gc, sort=False)[eo].first()

    tbl = grp_u.copy()
    tbl["EV_obs"] = eo_u

    if df_cal is None or cols_cal is None:
        tbl["EV_cal_mean"] = np.nan
        tbl["n_draws_cal"] = 0
        return tbl.reset_index()

    dc = df_cal.copy()
    ec = cols_cal["ev_cal"]
    if ec is None:
        raise ValueError("Calibrated dataframe missing EV_cal-like column.")
    dc["_ev_cal"] = pd.to_numeric(dc[ec], errors="coerce")
    grp_c = dc.groupby(gc, sort=False)["_ev_cal"].agg(["mean", "count"]).rename(
        columns={"mean": "EV_cal_mean", "count": "n_draws_cal"}
    )
    tbl = tbl.join(grp_c[["EV_cal_mean", "n_draws_cal"]], how="left")
    return tbl.reset_index()


def compute_mean_metrics(
    y_obs: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
) -> dict[str, float]:
    m: dict[str, float] = {}
    mask_before = np.isfinite(y_obs) & np.isfinite(pred_before)
    mask_after = np.isfinite(y_obs) & np.isfinite(pred_after)
    for name, ym, pp in ("before", mask_before, pred_before), ("after", mask_after, pred_after):
        if not np.any(ym):
            m[f"rmse_{name}"] = float("nan")
            m[f"mae_{name}"] = float("nan")
            m[f"bias_{name}"] = float("nan")
            m[f"median_ae_{name}"] = float("nan")
            m[f"r2_{name}"] = float("nan")
        else:
            yo, pr = y_obs[ym], pp[ym]
            m[f"rmse_{name}"] = float(np.sqrt(mean_squared_error(yo, pr)))
            m[f"mae_{name}"] = float(mean_absolute_error(yo, pr))
            m[f"bias_{name}"] = float(np.mean(pr - yo))
            m[f"median_ae_{name}"] = float(np.median(np.abs(pr - yo)))
            if yo.size >= 3 and np.var(yo) > 1e-12:
                m[f"r2_{name}"] = float(r2_score(yo, pr))
            else:
                m[f"r2_{name}"] = float("nan")
    return m


def _interval_cover_width(samples: np.ndarray, y_obs: float, alpha: float) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if len(s) < 2 or not np.isfinite(y_obs):
        return np.nan, np.nan
    q_lo = (1.0 - alpha) / 2.0
    q_hi = 1.0 - q_lo
    lo, hi = np.quantile(s, [q_lo, q_hi])
    width = float(hi - lo)
    cover = float(lo <= y_obs <= hi)
    return cover, width


def compute_interval_metrics(
    df_uncal: pd.DataFrame,
    df_cal: pd.DataFrame | None,
    cols_uncal: dict[str, str],
    cols_cal: dict[str, str] | None,
    levels: list[float],
    *,
    rng: np.random.Generator,
    crps_subsample: int = 2000,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, list[float], list[float]]:
    gc = cols_uncal["event_col"]
    ed = cols_uncal["ev_dec"]
    eo = cols_uncal["ev_obs"]

    interval_rows = []
    out_summary: dict[str, Any] = {}

    du = df_uncal.copy()
    du["_xd"] = pd.to_numeric(du[ed], errors="coerce")
    dc_df = df_cal.copy() if df_cal is not None and cols_cal and cols_cal.get("ev_cal") else None
    if dc_df is not None:
        dc_df = dc_df.copy()
        dc_df["_xc"] = pd.to_numeric(dc_df[cols_cal["ev_cal"]], errors="coerce")

    pit_plot_before: list[float] = []
    pit_plot_after: list[float] = []
    crps_b: list[float] = []
    crps_a: list[float] = []

    grouped_u = {str(k): v for k, v in du.groupby(gc, sort=False)}
    grouped_c = {str(k): v for k, v in dc_df.groupby(gc, sort=False)} if dc_df is not None else {}

    def sample_crps(s: np.ndarray, y: float) -> float:
        s = s[np.isfinite(s)]
        if len(s) < 2 or not np.isfinite(y):
            return float("nan")
        if len(s) > crps_subsample:
            idx = rng.choice(len(s), size=crps_subsample, replace=False)
            s = s[idx]
        e1 = float(np.mean(np.abs(s - y)))
        e2 = float(np.mean(np.abs(s.reshape(-1, 1) - s.reshape(1, -1))))
        return float(e1 - 0.5 * e2)

    def pit_prop(s: np.ndarray, y: float) -> float:
        s = s[np.isfinite(s)]
        if len(s) == 0 or not np.isfinite(y):
            return float("nan")
        return float(np.mean(s <= y))

    for eid, g in grouped_u.items():
        eid_s = str(eid)
        y = float(pd.to_numeric(g[eo].iloc[0], errors="coerce"))
        sb = pd.to_numeric(g["_xd"], errors="coerce").to_numpy(dtype=np.float64)
        crps_b.append(sample_crps(sb, y))

        pb = pit_prop(sb, y)
        pit_plot_before.append(pb)

        if grouped_c and eid_s in grouped_c:
            sc = pd.to_numeric(grouped_c[eid_s]["_xc"], errors="coerce").to_numpy(dtype=np.float64)
            pa = pit_prop(sc, y)
            pit_plot_after.append(pa)
            crps_a.append(sample_crps(sc, y))
            sc_full = sc
        elif not grouped_c:
            pit_plot_after.append(float("nan"))
            sc_full = np.array([], dtype=np.float64)
        else:
            pit_plot_after.append(float("nan"))
            sc_full = np.array([], dtype=np.float64)

        for alpha in levels:
            cov_b, w_b = _interval_cover_width(sb, y, alpha)
            if len(sc_full) >= 2:
                cov_a, w_a = _interval_cover_width(sc_full, y, alpha)
            else:
                cov_a, w_a = float("nan"), float("nan")
            interval_rows.append(
                {
                    "event_id": eid_s,
                    "nominal_coverage": alpha,
                    "coverage_before": float(cov_b) if np.isfinite(cov_b) else float("nan"),
                    "coverage_after": float(cov_a) if np.isfinite(cov_a) else float("nan"),
                    "width_before": float(w_b) if np.isfinite(w_b) else float("nan"),
                    "width_after": float(w_a) if np.isfinite(w_a) else float("nan"),
                }
            )

    def agg_interval(name_suffix: str, before_key: str, after_key: str) -> None:
        df_i = pd.DataFrame(interval_rows)
        for alpha in levels:
            sub = df_i[df_i["nominal_coverage"] == alpha]
            cob = pd.to_numeric(sub["coverage_before"], errors="coerce").mean(skipna=True)
            coa = pd.to_numeric(sub["coverage_after"], errors="coerce").mean(skipna=True)
            wmb = pd.to_numeric(sub["width_before"], errors="coerce").mean(skipna=True)
            wma = pd.to_numeric(sub["width_after"], errors="coerce").mean(skipna=True)
            wmed_b = pd.to_numeric(sub["width_before"], errors="coerce").median(skipna=True)
            wmed_a = pd.to_numeric(sub["width_after"], errors="coerce").median(skipna=True)
            key = str(alpha).replace(".", "_")
            out_summary[f"coverage_nominal_{key}"] = float(alpha)
            out_summary[f"empirical_coverage_before_{key}"] = float(cob)
            out_summary[f"empirical_coverage_after_{key}"] = float(coa) if np.isfinite(coa) else float("nan")
            out_summary[f"coverage_error_before_{key}"] = float(cob - alpha) if np.isfinite(cob) else float("nan")
            out_summary[f"coverage_error_after_{key}"] = float(coa - alpha) if np.isfinite(coa) else float("nan")
            out_summary[f"mean_width_before_{key}"] = float(wmb) if np.isfinite(wmb) else float("nan")
            out_summary[f"mean_width_after_{key}"] = float(wma) if np.isfinite(wma) else float("nan")
            out_summary[f"median_width_before_{key}"] = float(wmed_b) if np.isfinite(wmed_b) else float("nan")
            out_summary[f"median_width_after_{key}"] = float(wmed_a) if np.isfinite(wmed_a) else float("nan")

    agg_interval("", "", "")
    pb = pd.to_numeric(pd.Series(pit_plot_before), errors="coerce").dropna()
    pa = pd.to_numeric(pd.Series(pit_plot_after), errors="coerce").dropna()
    out_summary["pit_summary_before"] = {
        "mean": float(pb.mean()),
        "std": float(pb.std(ddof=0)),
        "frac_below_0.05": float(np.mean(pb < 0.05)),
        "frac_above_0.95": float(np.mean(pb > 0.95)),
        "histogram_10_bins": np.histogram(pb, bins=10, range=(0, 1))[0].astype(int).tolist(),
    }
    out_summary["pit_summary_after"] = (
        {
            "mean": float(pa.mean()),
            "std": float(pa.std(ddof=0)),
            "frac_below_0.05": float(np.mean(pa < 0.05)),
            "frac_above_0.95": float(np.mean(pa > 0.95)),
            "histogram_10_bins": np.histogram(pa, bins=10, range=(0, 1))[0].astype(int).tolist(),
        }
        if len(pa)
        else {}
    )
    cb = pd.to_numeric(pd.Series(crps_b), errors="coerce").mean(skipna=True)
    ca = pd.to_numeric(pd.Series(crps_a), errors="coerce").mean(skipna=True)
    out_summary["mean_crps_before"] = float(cb) if np.isfinite(cb) else float("nan")
    out_summary["mean_crps_after"] = float(ca) if np.isfinite(ca) else float("nan")
    if np.isfinite(cb) and np.isfinite(ca) and cb > 0:
        out_summary["crps_pct_improvement"] = float(100.0 * (cb - ca) / cb)
    else:
        out_summary["crps_pct_improvement"] = float("nan")

    cov_tab = pd.DataFrame(interval_rows)
    cov_agg_list = []
    alphas_unique = sorted({float(r["nominal_coverage"]) for r in interval_rows}, key=float) if interval_rows else []
    for alpha in alphas_unique:
        sub = cov_tab[cov_tab["nominal_coverage"] == alpha]
        if sub.empty:
            continue
        cob = pd.to_numeric(sub["coverage_before"], errors="coerce").mean()
        coa = pd.to_numeric(sub["coverage_after"], errors="coerce").mean()
        wmb = pd.to_numeric(sub["width_before"], errors="coerce").mean()
        wma = pd.to_numeric(sub["width_after"], errors="coerce").mean()
        cov_agg_list.append(
            {
                "nominal_coverage": float(alpha),
                "empirical_coverage_before": float(cob),
                "empirical_coverage_after": float(coa) if np.isfinite(coa) else float("nan"),
                "coverage_error_before": float(cob - alpha) if np.isfinite(cob) else float("nan"),
                "coverage_error_after": float(coa - alpha) if np.isfinite(coa) else float("nan"),
                "mean_width_before": float(wmb) if np.isfinite(wmb) else float("nan"),
                "mean_width_after": float(wma) if np.isfinite(wma) else float("nan"),
                "median_width_before": float(pd.to_numeric(sub["width_before"], errors="coerce").median()),
                "median_width_after": float(pd.to_numeric(sub["width_after"], errors="coerce").median()),
            }
        )
    pit_table = pd.DataFrame(
        {
            "pit_before_mean": pb.mean(),
            "pit_before_std": pb.std(ddof=0),
            "pit_before_frac_lt_05": np.mean(pb < 0.05),
            "pit_before_frac_gt_95": np.mean(pb > 0.95),
            "pit_after_mean": pa.mean() if len(pa) else np.nan,
            "pit_after_std": pa.std(ddof=0) if len(pa) else np.nan,
            "pit_after_frac_lt_05": np.mean(pa < 0.05) if len(pa) else np.nan,
            "pit_after_frac_gt_95": np.mean(pa > 0.95) if len(pa) else np.nan,
        },
        index=[0],
    )
    return out_summary, cov_agg_list, pit_table, pit_plot_before, pit_plot_after


def compute_pitch_group_diagnostics(
    df_uncal: pd.DataFrame,
    df_cal: pd.DataFrame | None,
    cols_uncal: dict[str, str],
    cols_cal: dict[str, str] | None,
) -> pd.DataFrame | None:
    col = None
    for c in ("pitch_group", "pitch_type"):
        if c in df_uncal.columns:
            col = c
            break
    if col is None:
        return None
    gc = cols_uncal["event_col"]
    ed = cols_uncal["ev_dec"]
    eo = cols_uncal["ev_obs"]

    dc = df_cal.copy() if df_cal is not None and cols_cal and cols_cal.get("ev_cal") else None
    if dc is None:
        return None
    dc = dc.copy()
    dc["_xc"] = pd.to_numeric(dc[cols_cal["ev_cal"]], errors="coerce")

    merge_key = gc
    rows = []
    for pg, gid in df_uncal.groupby(col, dropna=False):
        ids = gid[merge_key].unique()
        errs_b: list[float] = []
        errs_a: list[float] = []
        bias_b: list[float] = []
        bias_a: list[float] = []
        p99_b: list[float] = []
        p99_a: list[float] = []
        frac110_b: list[float] = []
        frac110_a: list[float] = []
        obs110: list[float] = []

        for eid in ids:
            g_b = gid[gid[merge_key].eq(eid)]
            ys = float(pd.to_numeric(g_b[eo].iloc[0], errors="coerce"))

            xd = pd.to_numeric(g_b[ed], errors="coerce").to_numpy()
            xd = xd[np.isfinite(xd)]

            gm_b = np.mean(xd) if len(xd) else float("nan")
            errs_b.append(abs(ys - gm_b))
            bias_b.append(gm_b - ys)
            if len(xd):
                p99_b.append(np.quantile(xd, 0.99))
                frac110_b.append(float(np.mean(xd > 110)))
            obs110.append(float(ys > 110))

            g_c = dc[dc[merge_key].eq(eid)] if dc is not None else None
            if g_c is not None and len(g_c):
                xc = pd.to_numeric(g_c["_xc"], errors="coerce").to_numpy()
                xc = xc[np.isfinite(xc)]
                gm_a = float(np.mean(xc)) if len(xc) else float("nan")
                errs_a.append(abs(ys - gm_a))
                bias_a.append(gm_a - ys)
                if len(xc):
                    p99_a.append(np.quantile(xc, 0.99))
                    frac110_a.append(float(np.mean(xc > 110)))

        def _msafe(xs: list[float]) -> float:
            return float(np.nanmean(xs)) if xs else float("nan")

        row = {
            "pitch_group": str(pg),
            "n_events": int(len(ids)),
            "rmse_before": float(np.sqrt(np.nanmean(np.square(np.array(errs_b))))) if errs_b else float("nan"),
            "rmse_after": float(np.sqrt(np.nanmean(np.square(np.array(errs_a))))) if errs_a else float("nan"),
            "bias_before": _msafe(bias_b),
            "bias_after": _msafe(bias_a),
            "p99_draw_ev_before": _msafe(p99_b),
            "p99_draw_ev_after": _msafe(p99_a),
            "p_draw_gt_110_before": _msafe(frac110_b),
            "p_draw_gt_110_after": _msafe(frac110_a),
            "observed_frac_ev_gt_110": float(np.mean(obs110)) if obs110 else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_tail_diagnostics(
    df_uncal: pd.DataFrame,
    df_cal: pd.DataFrame | None,
    cols_uncal: dict[str, str],
    cols_cal: dict[str, str] | None,
    thresholds: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    ed = cols_uncal["ev_dec"]
    gc = cols_uncal["event_col"]
    eo = cols_uncal["ev_obs"]

    xb = pd.to_numeric(df_uncal[ed], errors="coerce").to_numpy(dtype=np.float64)
    xb = xb[np.isfinite(xb)]
    xa = np.array([], dtype=np.float64)
    if df_cal is not None and cols_cal and cols_cal.get("ev_cal"):
        xa = pd.to_numeric(df_cal[cols_cal["ev_cal"]], errors="coerce").to_numpy(dtype=np.float64)
        xa = xa[np.isfinite(xa)]

    def tail_stats(x: np.ndarray) -> dict[str, float]:
        if len(x) == 0:
            return {k: float("nan") for k in ("max", "p95", "p99", "p995", "p999")}
        return {
            "max": float(np.max(x)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "p995": float(np.quantile(x, 0.995)),
            "p999": float(np.quantile(x, 0.999)),
        }

    draw_summary = {
        "draw_level_before": tail_stats(xb),
        "draw_level_after": tail_stats(xa) if len(xa) else {},
    }
    for t in thresholds:
        draw_summary[f"frac_draws_gt_{t}_before"] = float(np.mean(xb > t)) if len(xb) else float("nan")
        draw_summary[f"frac_draws_gt_{t}_after"] = float(np.mean(xa > t)) if len(xa) else float("nan")

    ev_obs_by = df_uncal.groupby(gc, sort=False)[eo].first()
    y_obs = pd.to_numeric(ev_obs_by, errors="coerce").to_numpy(dtype=np.float64)
    y_obs = y_obs[np.isfinite(y_obs)]

    rows = []
    for t in thresholds:
        obs_rate = float(np.mean(y_obs > t)) if len(y_obs) else float("nan")
        pred_b = float(np.mean(xb > t)) if len(xb) else float("nan")
        pred_a = float(np.mean(xa > t)) if len(xa) else float("nan")
        rows.append(
            {
                "threshold": t,
                "observed_exceedance_rate": obs_rate,
                "pred_exceedance_before": pred_b,
                "pred_exceedance_after": pred_a,
                "abs_error_before": abs(pred_b - obs_rate) if np.isfinite(pred_b) else float("nan"),
                "abs_error_after": abs(pred_a - obs_rate) if np.isfinite(pred_a) else float("nan"),
            }
        )
    return draw_summary, pd.DataFrame(rows)


def make_plots(
    event_tbl: pd.DataFrame,
    df_uncal: pd.DataFrame,
    df_cal: pd.DataFrame | None,
    cols_uncal: dict[str, str],
    cols_cal: dict[str, str] | None,
    pit_before: list[float],
    pit_after: list[float],
    tail_df: pd.DataFrame,
    output_plots: Path,
) -> None:
    output_plots.mkdir(parents=True, exist_ok=True)
    y_obs = event_tbl["EV_obs"].to_numpy(dtype=np.float64)
    y_dec = event_tbl["EV_dec_mean"].to_numpy(dtype=np.float64)
    y_cal = event_tbl["EV_cal_mean"].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_obs) & np.isfinite(y_dec)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_obs[finite], y_dec[finite], alpha=0.35, s=14)
    lo = float(np.nanmin(np.r_[y_obs[finite], y_dec[finite]])) - 2
    hi = float(np.nanmax(np.r_[y_obs[finite], y_dec[finite]])) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("EV_obs")
    ax.set_ylabel("EV_dec_mean")
    ax.set_title("Observed vs decoded predictive mean")
    fig.savefig(output_plots / "observed_vs_pred_mean_before.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fc = np.isfinite(y_obs) & np.isfinite(y_cal)
    fig, ax = plt.subplots(figsize=(6, 6))
    if np.any(fc):
        ax.scatter(y_obs[fc], y_cal[fc], alpha=0.35, s=14, color="darkgreen")
        lo2 = float(np.nanmin(np.r_[y_obs[fc], y_cal[fc]])) - 2
        hi2 = float(np.nanmax(np.r_[y_obs[fc], y_cal[fc]])) + 2
        ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=1)
    ax.set_xlabel("EV_obs")
    ax.set_ylabel("EV_cal_mean")
    ax.set_title("Observed vs calibrated predictive mean")
    fig.savefig(output_plots / "observed_vs_pred_mean_after.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    r_b = y_dec - y_obs
    r_a = y_cal - y_obs
    ax.hist(r_b[np.isfinite(r_b)], bins=45, density=True, alpha=0.5, label="EV_dec_mean - EV_obs")
    ax.hist(r_a[np.isfinite(r_a)], bins=45, density=True, alpha=0.5, label="EV_cal_mean - EV_obs")
    ax.legend()
    ax.set_title("Residuals before / after")
    fig.savefig(output_plots / "residual_hist_before_after.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    pb = np.asarray(pit_before, dtype=np.float64)
    pb = pb[np.isfinite(pb)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pb, bins=10, range=(0, 1), density=True, edgecolor="k")
    ax.set_title("PIT before")
    ax.set_xlabel("PIT proportion (draws ≤ obs)")
    fig.savefig(output_plots / "pit_hist_before.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    pa = np.asarray(pit_after, dtype=np.float64)
    pa = pa[np.isfinite(pa)]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(pa):
        ax.hist(pa, bins=10, range=(0, 1), density=True, color="green", edgecolor="k")
    ax.set_title("PIT after")
    ax.set_xlabel("PIT proportion (draws ≤ obs)")
    fig.savefig(output_plots / "pit_hist_after.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    ed = cols_uncal["ev_dec"]
    xb_all = pd.to_numeric(df_uncal[ed], errors="coerce").to_numpy(dtype=np.float64)
    xb_all = xb_all[np.isfinite(xb_all)]
    xa_all = np.array([], dtype=np.float64)
    if df_cal is not None and cols_cal and cols_cal.get("ev_cal"):
        xa_all = pd.to_numeric(df_cal[cols_cal["ev_cal"]], errors="coerce").to_numpy(dtype=np.float64)
        xa_all = xa_all[np.isfinite(xa_all)]

    fig, ax = plt.subplots(figsize=(8, 4))
    pool = []
    if len(xb_all):
        pool.extend(xb_all[:200_000].tolist())
    if len(xa_all):
        pool.extend(xa_all[:200_000].tolist())
    ymax = float(np.percentile(pool, 99.9)) if pool else 120.0
    ymax = max(ymax, float(np.nanmax(y_obs[np.isfinite(y_obs)])) + 10, 115.0)
    loe = 40.0
    bins_ev = np.linspace(loe, ymax, min(100, max(40, len(xb_all) // 100 + 40)))
    if len(xb_all):
        ax.hist(xb_all, bins=bins_ev, density=True, alpha=0.45, label="uncalibrated draws")
    if len(xa_all):
        ax.hist(xa_all, bins=bins_ev, density=True, alpha=0.45, label="calibrated draws", color="green")
    ax.hist(y_obs[np.isfinite(y_obs)], bins=bins_ev, density=True, alpha=0.35, histtype="step", linewidth=2, label="observed EV (events)")
    ax.legend()
    ax.set_title("EV draw distributions vs observed")
    ax.set_xlabel("EV (mph)")
    fig.savefig(output_plots / "ev_draw_distribution_before_after.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    z = 95.0
    xbz = xb_all[xb_all >= z]
    xaz = xa_all[xa_all >= z] if len(xa_all) else np.array([], dtype=np.float64)
    yobsz = y_obs[y_obs >= z]
    parts_hi: list[np.ndarray] = []
    if len(xbz):
        parts_hi.append(xbz)
    if len(xaz):
        parts_hi.append(xaz)
    if len(yobsz):
        parts_hi.append(yobsz)
    if parts_hi:
        hi_z = float(np.max(np.concatenate(parts_hi)))
    else:
        hi_z = 120.0
    hi_z = max(hi_z, z + 1.0)
    bins_z = np.linspace(z, hi_z, 60)
    if len(xbz):
        ax.hist(xbz, bins=bins_z, density=True, alpha=0.45, label="uncal ≥95")
    if len(xaz):
        ax.hist(xaz, bins=bins_z, density=True, alpha=0.45, label="cal ≥95", color="green")
    ax.legend()
    ax.set_title("Tail zoom EV ≥ 95 mph")
    fig.savefig(output_plots / "ev_tail_zoom_before_after.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    obs_s = np.sort(np.asarray(y_obs[np.isfinite(y_obs)], dtype=np.float64))
    dec_s = np.sort(np.asarray(y_dec[np.isfinite(y_dec)], dtype=np.float64))
    cal_s = np.sort(np.asarray(y_cal[np.isfinite(y_cal)], dtype=np.float64))
    n = min(len(obs_s), len(dec_s), len(cal_s))
    if n >= 10:
        fig, ax = plt.subplots(figsize=(7, 6))
        xq = (np.arange(n) + 0.5) / n
        ax.plot(xq, obs_s[:n], label="sorted EV_obs", lw=1.6)
        ax.plot(xq, dec_s[:n], label="sorted EV_dec_mean", alpha=0.8)
        ax.plot(xq, cal_s[:n], label="sorted EV_cal_mean", color="green", alpha=0.8)
        ax.set_xlabel("empirical quantile rank")
        ax.set_ylabel("mph")
        ax.legend()
        ax.set_title("QQ-style: sorted event-level means")
        fig.savefig(output_plots / "qq_pred_mean_before_after.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    if len(tail_df):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tail_df["threshold"], tail_df["observed_exceedance_rate"], "ko-", label="observed rate")
        ax.plot(tail_df["threshold"], tail_df["pred_exceedance_before"], "bs-", alpha=0.7, label="pred uncalibrated")
        ax.plot(tail_df["threshold"], tail_df["pred_exceedance_after"], "g^-", alpha=0.7, label="pred calibrated")
        ax.set_xlabel("Threshold (mph)")
        ax.set_ylabel("Exceedance rate")
        ax.legend()
        ax.set_title("Tail exceedance: observed vs predicted")
        fig.savefig(output_plots / "tail_exceedance_before_after.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncalibrated-draws-path", type=Path, required=True)
    ap.add_argument("--calibrated-draws-path", type=Path, default=None)
    ap.add_argument("--calibrator-path", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--event-id-col", type=str, default="event_id")
    ap.add_argument("--ev-dec-col", type=str, default="EV_dec")
    ap.add_argument("--ev-cal-col", type=str, default="EV_cal")
    ap.add_argument("--ev-obs-col", type=str, default="EV_obs")
    ap.add_argument(
        "--interval-levels",
        type=float,
        nargs="+",
        default=[0.50, 0.80, 0.90],
    )
    ap.add_argument(
        "--tail-thresholds",
        type=float,
        nargs="+",
        default=[100.0, 105.0, 110.0, 115.0, 120.0],
    )
    ap.add_argument("--crps-seed", type=int, default=42)
    ap.add_argument("--crps-subsample", type=int, default=2000)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_root = Path(args.output_dir).resolve() if args.output_dir else REPO / "artifacts" / "physics_decoder_calibration_diagnostics" / stamp
    plots_dir = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_u = load_draws(Path(args.uncalibrated_draws_path))
    df_u, cols_u = infer_columns(
        df_u,
        event_id_col_hint=args.event_id_col if args.event_id_col in df_u.columns else None,
        ev_dec_candidates=[
            args.ev_dec_col,
            "EV_dec",
            "ev_dec",
            "EV",
            "launch_speed_dec",
            "launch_speed_sim",
        ],
        ev_cal_candidates=["EV_cal"],
        ev_obs_candidates=[
            args.ev_obs_col,
            "EV_obs",
            "ev_obs",
            "observed_EV",
            "launch_speed",
        ],
    )

    df_c: pd.DataFrame | None = None
    cols_c: dict[str, str] | None = None
    calibrator_used = False

    if args.calibrated_draws_path is not None:
        df_raw = load_draws(Path(args.calibrated_draws_path))
        df_c, cols_c = infer_columns(
            df_raw,
            event_id_col_hint=args.event_id_col if args.event_id_col in df_raw.columns else None,
            ev_dec_candidates=[args.ev_dec_col, "EV_dec", "ev_dec", "EV"],
            ev_cal_candidates=[args.ev_cal_col, "EV_cal", "ev_cal", "calibrated_EV"],
            ev_obs_candidates=[args.ev_obs_col, "EV_obs", "observed_EV", "launch_speed"],
        )
        if cols_c["event_col"] != cols_u["event_col"]:
            raise ValueError("Event id alignment mismatch between uncalibrated and calibrated draws.")
    elif args.calibrator_path:
        from sbi_forward_sim.src.physics_calibration import EVPhysicsCalibrator

        model = EVPhysicsCalibrator.load(Path(args.calibrator_path))
        dc_ev = getattr(model, "ev_col", "EV")
        df_c = df_u.copy()
        if dc_ev not in df_c.columns:
            df_c = df_c.copy()
            df_c[dc_ev] = pd.to_numeric(df_c[cols_u["ev_dec"]], errors="coerce")
        df_c = model.transform_draws(df_c, group_col=cols_u["event_col"])
        if "EV_cal" not in df_c.columns:
            raise RuntimeError("Calibrator did not add EV_cal")
        cols_c = {
            "event_col": cols_u["event_col"],
            "ev_dec": cols_u["ev_dec"],
            "ev_cal": "EV_cal",
            "ev_obs": cols_u["ev_obs"],
        }
        calibrator_used = True
    else:
        raise ValueError("Provide either --calibrated-draws-path or --calibrator-path.")

    rng = np.random.default_rng(args.crps_seed)
    assert cols_c is not None
    ekey = cols_u["event_col"]

    def _norm_event_ids(s: pd.Series) -> pd.Series:
        out: list[str] = []
        for v in s.tolist():
            if pd.isna(v):
                out.append("__na__")
                continue
            if isinstance(v, (np.integer, int)):
                out.append(str(int(v)))
                continue
            t = str(v).strip()
            if t.endswith(".0") and t[:-2].lstrip("-").isdigit():
                t = t[:-2]
            out.append(t)
        return pd.Series(out, index=s.index, dtype=str)

    df_u[ekey] = _norm_event_ids(df_u[ekey])
    df_c[ekey] = _norm_event_ids(df_c[ekey])

    if cols_c.get("ev_cal") and cols_c["ev_cal"] != "EV_cal" and "EV_cal" not in df_c.columns:
        df_c = df_c.copy()
        df_c["EV_cal"] = df_c[cols_c["ev_cal"]]
        cols_c = {**cols_c, "ev_cal": "EV_cal"}

    event_tbl_raw = build_event_level_table(df_u, df_c, cols_u, cols_c)
    event_col_actual = cols_u["event_col"]

    event_tbl = event_tbl_raw.rename(columns={event_col_actual: "event_key"})

    y_obs_arr = pd.to_numeric(event_tbl["EV_obs"], errors="coerce").to_numpy(dtype=np.float64)
    y_dec_mean = pd.to_numeric(event_tbl["EV_dec_mean"], errors="coerce").to_numpy(dtype=np.float64)
    y_cal_mean = pd.to_numeric(event_tbl["EV_cal_mean"], errors="coerce").to_numpy(dtype=np.float64)

    mean_met = compute_mean_metrics(y_obs_arr, y_dec_mean, y_cal_mean)
    tail_summ, tail_df = compute_tail_diagnostics(df_u, df_c, cols_u, cols_c, list(args.tail_thresholds))

    int_summ, cov_rows, pit_one_row, pit_before_list, pit_after_list = compute_interval_metrics(
        df_u,
        df_c,
        cols_u,
        cols_c,
        list(args.interval_levels),
        rng=rng,
        crps_subsample=args.crps_subsample,
    )

    grp_df = compute_pitch_group_diagnostics(df_u, df_c, cols_u, cols_c)

    make_plots(
        pd.DataFrame({"EV_obs": event_tbl["EV_obs"], "EV_dec_mean": event_tbl["EV_dec_mean"], "EV_cal_mean": event_tbl["EV_cal_mean"]}),
        df_u,
        df_c,
        cols_u,
        cols_c,
        pit_before_list,
        pit_after_list,
        tail_df,
        plots_dir,
    )

    evt_out = pd.DataFrame(event_tbl_raw)
    if "event_col" not in evt_out.columns:
        evt_out = evt_out.rename(columns={evt_out.columns[0]: "event_id"})
    evt_out.to_csv(out_root / "event_level_diagnostics.csv", index=False)

    cov_df_out = pd.DataFrame(cov_rows)
    cov_df_out.to_csv(out_root / "interval_coverage_table.csv", index=False)

    pit_one_row.to_csv(out_root / "pit_summary_table.csv", index=False)
    tail_df.to_csv(out_root / "tail_exceedance_table.csv", index=False)
    if grp_df is not None and not grp_df.empty:
        grp_df.to_csv(out_root / "pitch_group_diagnostics.csv", index=False)

    metrics_summary = {
        "mean_metrics": mean_met,
        "interval_and_crps": int_summ,
        "tail_draw_level": tail_summ,
        "calibrator_applied_inside_script": calibrator_used,
        "paths": {"uncalibrated": str(Path(args.uncalibrated_draws_path)), "calibrated": str(args.calibrated_draws_path), "joblib": str(args.calibrator_path)},
    }

    json_ready = sanitize_for_json(metrics_summary)
    (out_root / "metrics_summary.json").write_text(json.dumps(json_ready, indent=2), encoding="utf-8")

    k90 = None
    for alpha in sorted(args.interval_levels, reverse=True):
        if alpha >= 0.89:
            k90 = str(alpha).replace(".", "_")
            break

    rmse_b = mean_met["rmse_before"]
    rmse_a = mean_met["rmse_after"]
    cov90_b = float(int_summ[f"empirical_coverage_before_{k90}"]) if k90 and f"empirical_coverage_before_{k90}" in int_summ else float("nan")
    cov90_a = float(int_summ[f"empirical_coverage_after_{k90}"]) if k90 and f"empirical_coverage_after_{k90}" in int_summ else float("nan")

    print("\nHeld-out EV Calibration Diagnostics\n-----------------------------------")
    print(f"Mean RMSE before: {mean_met['rmse_before']:.4f}")
    print(f"Mean RMSE after:  {mean_met['rmse_after']:.4f}")
    print(f"Mean bias before: {mean_met['bias_before']:.4f}")
    print(f"Mean bias after:  {mean_met['bias_after']:.4f}")
    print(f"Mean CRPS before: {int_summ.get('mean_crps_before')}")
    print(f"Mean CRPS after:  {int_summ.get('mean_crps_after')}")
    if k90 and np.isfinite(cov90_b):
        print(f"90% coverage before: {cov90_b:.4f}")
        print(f"90% coverage after: {cov90_a:.4f}")
    thresh_115 = 115.0
    if thresh_115 in args.tail_thresholds:
        fr_b = tail_summ.get(f"frac_draws_gt_{thresh_115}_before", float("nan"))
        fr_a = tail_summ.get(f"frac_draws_gt_{thresh_115}_after", float("nan"))
        idx = tail_df[np.isclose(tail_df["threshold"].astype(np.float64), thresh_115)] if len(tail_df) else None
        obs115 = float(idx["observed_exceedance_rate"].iloc[0]) if idx is not None and len(idx) else float("nan")
        print(f"P(draw EV > 115) before: {fr_b:.4f}" if np.isfinite(fr_b) else "P(draw EV > 115) before: nan")
        print(f"P(draw EV > 115) after:  {fr_a:.4f}" if np.isfinite(fr_a) else "P(draw EV > 115) after: nan")
        print(f"Observed P(EV > 115):    {obs115:.4f}")
    print(f"Max draw EV before: {tail_summ['draw_level_before'].get('max')}")
    da = tail_summ.get("draw_level_after", {})
    print(f"Max draw EV after:  {da.get('max')}")

    print("\n--- Interpretation (automated, not authoritative) ---\n")

    errs_b_tail = tail_df["abs_error_before"].tolist() if len(tail_df) else []
    errs_a_tail = tail_df["abs_error_after"].tolist() if len(tail_df) else []
    tail_avg_improve = bool(errs_b_tail and errs_a_tail and np.nanmean(errs_a_tail) < np.nanmean(errs_b_tail))

    rmse_improved = np.isfinite(rmse_b) and np.isfinite(rmse_a) and rmse_a < rmse_b
    bias_cls = np.isfinite(mean_met["bias_before"]) and np.isfinite(mean_met["bias_after"]) and (
        abs(mean_met["bias_after"]) < abs(mean_met["bias_before"])
    )

    cov_worse = False
    if k90 and np.isfinite(cov90_b) and np.isfinite(cov90_a):
        nom_m = [a for a in args.interval_levels if str(a).replace(".", "_") == k90]
        nom = float(nom_m[0]) if nom_m else float("nan")
        err_b = cov90_b - nom
        err_a = cov90_a - nom
        cov_worse = abs(err_a) > abs(err_b) + 0.02

    print(f"- RMSE: {'improved' if rmse_improved else 'did not improve' if np.isfinite(rmse_a) else 'n/a'} (before {mean_met['rmse_before']:.4f}, after {mean_met['rmse_after']:.4f}).")
    print(f"- Bias vs obs: {'moved closer to zero' if bias_cls else 'did not consistently move closer to zero'}.")
    print(f"- Tail exceedance errors (pool): {'smaller after cal on avg' if tail_avg_improve else 'mixed or unchanged'}.")
    if k90 and np.isfinite(cov90_a):
        print(f"- {_fmt_cov_line(cov90_b, cov90_a, cov_worse)}")
    if rmse_improved and cov_worse:
        print("- WARNING: Point accuracy improved but interval coverage degraded — check probabilistic honesty.")
    if np.isfinite(tail_summ.get("draw_level_after", {}).get("max", np.nan)):
        frac120_b = tail_summ.get("frac_draws_gt_120.0_before", np.nan)
        frac120_a = tail_summ.get("frac_draws_gt_120.0_after", np.nan)
        if np.isfinite(frac120_b) and np.isfinite(frac120_a) and frac120_a < 0.5 * frac120_b and frac120_b > 0.001:
            print("- WARNING: Calibrated draws show much smaller mass beyond 120 mph than uncalibrated; confirm if tail suppression is intended.")

    print(f"\nWrote diagnostics to {out_root}\n")


def _fmt_cov_line(cov_b: float, cov_a: float, cov_worse: bool) -> str:
    s = (
        f"Nominal-interval coverage vs target: empirical rates before vs after "
        f"({cov_b:.3f}, {cov_a:.3f}). {'Coverage calibration somewhat worse.' if cov_worse else ''}"
    )
    return s


if __name__ == "__main__":
    main()
