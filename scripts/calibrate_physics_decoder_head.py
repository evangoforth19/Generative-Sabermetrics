#!/usr/bin/env python3
"""
Fit EV-only post-hoc calibration for physics decoder outputs from held-out pipeline draws.

Default (--split-mode baseline): fit on all official train+calibration events (~5644) using EV_obs
from baseline_direct_y_*.parquet; report RMSE / calibration metrics on the official test set (1010).
Requires predictive draws covering those events (see --fit-heldout-dir / --eval-heldout-dir / --heldout-dir).

Legacy: --split-mode random for 60/20/20 on a single run.

Consumes outputs from run_heldout_pipeline_test.py (predictive_draws_*.parquet/csv).
Does not retrain stage-u or stage-z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MMC2 = REPO / "MCMC 2"
SBI = MMC2 / "sbi_forward_sim"
if str(MMC2) not in sys.path:
    sys.path.insert(0, str(MMC2))

from sbi_forward_sim.src.physics_calibration import (  # noqa: E402
    CONTEXT_COLUMNS,
    EVPhysicsCalibrator,
    merge_context_on_events,
    soft_cap_ev,
    summarize_ev_per_event,
)

STAT_COLS = [
    "EV_dec_mean",
    "EV_dec_std",
    "EV_dec_p05",
    "EV_dec_p10",
    "EV_dec_p25",
    "EV_dec_p50",
    "EV_dec_p75",
    "EV_dec_p90",
    "EV_dec_p95",
    "EV_dec_p99",
    "n_draws",
]


def _resolve_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not resolve {label}; tried {candidates}. Columns: {list(df.columns)[:60]}")


def _find_draws_file(heldout_dir: Path) -> Path:
    for name in (
        "predictive_draws_admissible.parquet",
        "predictive_draws_admissible.csv",
        "predictive_draws.parquet",
        "predictive_draws.csv",
    ):
        p = heldout_dir / name
        if p.is_file():
            return p
    globs = list(heldout_dir.glob("predictive_draws*.parquet")) + list(heldout_dir.glob("predictive_draws*.csv"))
    if not globs:
        raise FileNotFoundError(f"No predictive_draws* in {heldout_dir}")
    return sorted(globs, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_baseline_split_ids(splits_dir: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    """Official train / calibration / test event_id sets (disjoint)."""
    d = Path(splits_dir)
    paths = {
        "train": d / "baseline_direct_y_train.parquet",
        "calibration": d / "baseline_direct_y_calibration.parquet",
        "test": d / "baseline_direct_y_test.parquet",
    }
    for k, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing {k} split: {p}")
    st = {str(x) for x in pd.read_parquet(paths["train"], columns=["event_id"])["event_id"].to_numpy()}
    sc = {str(x) for x in pd.read_parquet(paths["calibration"], columns=["event_id"])["event_id"].to_numpy()}
    se = {str(x) for x in pd.read_parquet(paths["test"], columns=["event_id"])["event_id"].to_numpy()}
    inter = (st & sc) | (st & se) | (sc & se)
    if inter:
        raise ValueError(f"baseline_direct_y splits overlap ({len(inter)} event_ids); check {splits_dir}")
    train_cal = st | sc
    return st, sc, se, train_cal


def _observed_ev_from_baseline(splits_dir: Path) -> dict[str, float]:
    """Ground-truth EV_obs from baseline_direct_y tables (train+cal+test)."""
    d = Path(splits_dir)
    parts: list[pd.DataFrame] = []
    for name in ("baseline_direct_y_train.parquet", "baseline_direct_y_calibration.parquet", "baseline_direct_y_test.parquet"):
        p = d / name
        parts.append(pd.read_parquet(p, columns=["event_id", "EV"]))
    all_df = pd.concat(parts, ignore_index=True)
    return {
        str(k): float(v)
        for k, v in zip(all_df["event_id"].to_numpy(), pd.to_numeric(all_df["EV"], errors="coerce").to_numpy())
        if np.isfinite(v)
    }


def _ensure_event_id_column(draws: pd.DataFrame, group_col: str) -> None:
    if group_col in draws.columns:
        return
    if all(c in draws.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
        draws[group_col] = (
            draws["game_pk"].astype(str)
            + "_"
            + draws["at_bat_number"].astype(str)
            + "_"
            + draws["pitch_number"].astype(str)
        )
        return
    raise ValueError(f"Need {group_col} or game_pk/at_bat_number/pitch_number; have {list(draws.columns)[:30]}")


def _load_draws_from_run(heldout_dir: Path, group_col: str) -> tuple[pd.DataFrame, str, Path]:
    draws_path = _find_draws_file(Path(heldout_dir))
    draws = _read_table(draws_path)
    ev_col = _resolve_col(draws, ["EV", "EV_dec", "ev_dec", "ev_mph"], "decoder EV")
    _ensure_event_id_column(draws, group_col)
    return draws, ev_col, draws_path


def _merge_run_summary_obs(heldout_dir: Path, group_col: str, obs_by_event: dict[str, float]) -> dict[str, float]:
    """Augment obs map from predictive_summary_by_event if present (random-split mode)."""
    out = dict(obs_by_event)
    summary_path = Path(heldout_dir) / "predictive_summary_by_event.parquet"
    if not summary_path.is_file():
        summary_path = Path(heldout_dir) / "predictive_summary_by_event.csv"
    if not summary_path.is_file():
        return out
    summ0 = _read_table(summary_path)
    eid_col = _resolve_col(summ0, ["event_id", "Event_ID"], "event id")
    oc = _resolve_col(summ0, ["observed_EV", "EV_obs", "launch_speed"], "observed EV")
    for k, v in zip(summ0[eid_col].to_numpy(), pd.to_numeric(summ0[oc], errors="coerce").to_numpy()):
        if np.isfinite(v):
            out[str(k)] = float(v)
    return out


def _merge_draws_column_obs(draws: pd.DataFrame, group_col: str, obs_by_event: dict[str, float]) -> dict[str, float]:
    out = dict(obs_by_event)
    for c in ("EV_obs", "observed_EV", "launch_speed"):
        if c not in draws.columns:
            continue
        tmp = draws[[group_col, c]].drop_duplicates(subset=[group_col])
        for k, v in zip(tmp[group_col].to_numpy(), pd.to_numeric(tmp[c], errors="coerce").to_numpy()):
            if np.isfinite(v):
                out[str(k)] = float(v)
    return out


def _build_event_summary_table(
    draws: pd.DataFrame,
    ev_col: str,
    group_col: str,
    obs_by_event: dict[str, float],
    context_master: Path | None,
) -> pd.DataFrame:
    summ = summarize_ev_per_event(draws, ev_col, group_col=group_col)
    summ["EV_obs"] = summ[group_col].astype(str).map(obs_by_event)
    summ = summ.dropna(subset=["EV_obs"]).copy()
    # Canonical residual convention:
    # EV_resid = EV_obs - EV_dec_mean
    summ["EV_resid"] = summ["EV_obs"].to_numpy(dtype=np.float64) - summ["EV_dec_mean"].to_numpy(dtype=np.float64)
    if context_master and Path(context_master).is_file() and pd.api.types.is_integer_dtype(summ[group_col]):
        master = pd.read_parquet(Path(context_master))
        if "spin_rate" not in master.columns and "release_spin_rate" in master.columns:
            master = master.copy()
            master["spin_rate"] = pd.to_numeric(master["release_spin_rate"], errors="coerce")
        if "pitcher_hand" not in master.columns and "p_throws" in master.columns:
            master["pitcher_hand"] = master["p_throws"]
        if "batter_hand" not in master.columns and "stand" in master.columns:
            master["batter_hand"] = master["stand"]
        summ_i = summ.copy()
        summ_i[group_col] = summ_i[group_col].astype(np.int64)
        summ = merge_context_on_events(summ_i, master, event_key=group_col)
    return summ


def _resolve_feature_lists(summ: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_extra: list[str] = []
    categorical_extra: list[str] = []
    for c in CONTEXT_COLUMNS:
        if c not in summ.columns:
            continue
        if pd.api.types.is_numeric_dtype(summ[c]):
            numeric_extra.append(c)
        else:
            categorical_extra.append(c)
    numeric_features = STAT_COLS + [c for c in numeric_extra if c not in STAT_COLS]
    categorical_features = list(dict.fromkeys(categorical_extra))
    return numeric_features, categorical_features


def _assert_residual_convention(df: pd.DataFrame, *, label: str) -> None:
    ev_obs = pd.to_numeric(df["EV_obs"], errors="coerce").to_numpy(dtype=np.float64)
    ev_dec = pd.to_numeric(df["EV_dec_mean"], errors="coerce").to_numpy(dtype=np.float64)
    resid = pd.to_numeric(df["EV_resid"], errors="coerce").to_numpy(dtype=np.float64)
    before_error = ev_dec - ev_obs

    mean_resid = float(np.nanmean(resid))
    mean_true = float(np.nanmean(ev_obs - ev_dec))
    mean_before = float(np.nanmean(before_error))

    if not np.isfinite(mean_resid) or not np.isfinite(mean_true):
        raise AssertionError(f"[{label}] residual means not finite: EV_resid={mean_resid}, true={mean_true}")
    if abs(mean_resid - mean_true) > 1e-8:
        raise AssertionError(
            f"[{label}] EV_resid mean mismatch: EV_resid={mean_resid:.10f}, EV_obs-EV_dec_mean={mean_true:.10f}"
        )
    if abs(mean_resid + mean_before) > 1e-8:
        raise AssertionError(
            f"[{label}] EV_resid mean should be -mean(before_error): "
            f"EV_resid={mean_resid:.10f}, before_error={mean_before:.10f}"
        )


def _log_pre_fit_residual_stats(df: pd.DataFrame, *, label: str) -> None:
    ev_obs = pd.to_numeric(df["EV_obs"], errors="coerce").to_numpy(dtype=np.float64)
    ev_dec = pd.to_numeric(df["EV_dec_mean"], errors="coerce").to_numpy(dtype=np.float64)
    resid = pd.to_numeric(df["EV_resid"], errors="coerce").to_numpy(dtype=np.float64)
    print(
        f"[{label}] pre-fit residual stats: "
        f"mean(EV_obs)={np.nanmean(ev_obs):.6f}, "
        f"mean(EV_dec_mean)={np.nanmean(ev_dec):.6f}, "
        f"mean(EV_resid)={np.nanmean(resid):.6f}, "
        f"median(EV_resid)={np.nanmedian(resid):.6f}, "
        f"share(EV_resid>0)={np.nanmean(resid > 0):.6f}, "
        f"share(EV_resid<0)={np.nanmean(resid < 0):.6f}"
    )


def _log_post_fit_residual_diagnostics(df: pd.DataFrame, *, label: str) -> None:
    true_resid = pd.to_numeric(df["EV_obs"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(
        df["EV_dec_mean"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    pred_resid = pd.to_numeric(df["pred_EV_resid"], errors="coerce").to_numpy(dtype=np.float64)
    before_error = pd.to_numeric(df["EV_dec_mean"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(
        df["EV_obs"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    after_error = pd.to_numeric(df["EV_cal_mean"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(
        df["EV_obs"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    correction = pd.to_numeric(df["EV_cal_mean"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(
        df["EV_dec_mean"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    corr_val = float(np.corrcoef(pred_resid, true_resid)[0, 1]) if len(df) > 2 else float("nan")
    print(
        f"[{label}] post-fit diagnostics: "
        f"mean(predicted_residual)={np.nanmean(pred_resid):.6f}, "
        f"corr(predicted_residual,true_residual)={corr_val:.6f}, "
        f"mean(EV_cal_mean-EV_dec_mean)={np.nanmean(correction):.6f}, "
        f"mean(EV_dec_mean-EV_obs)={np.nanmean(before_error):.6f}, "
        f"mean(EV_cal_mean-EV_obs)={np.nanmean(after_error):.6f}"
    )
    if np.nanmean(before_error) > 0 and np.nanmean(pred_resid) > 0:
        print(
            "WARNING: decoder overpredicts but calibrator correction is positive. "
            "This likely indicates wrong residual sign or failed residual learning.",
            file=sys.stderr,
        )


def _empirical_crps(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n < 2:
        return float("nan")
    e1 = float(np.mean(np.abs(s - y)))
    e2 = float(np.mean(np.abs(s.reshape(-1, 1) - s.reshape(1, -1))))
    return float(e1 - 0.5 * e2)


def _pit_rank(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n == 0:
        return float("nan")
    r = float(np.sum(s < y) + 0.5 * np.sum(s == y))
    return float(np.clip(r / (n + 1), 1e-6, 1 - 1e-6))


def _interval_stats(samples: np.ndarray, y: float, qlo: float, qhi: float) -> tuple[bool, float]:
    s = np.asarray(samples, dtype=np.float64).ravel()
    if len(s) < 2:
        return False, float("nan")
    lo, hi = np.quantile(s, [qlo, qhi])
    return bool(lo <= y <= hi), float(hi - lo)


def _tail_stats(samples: np.ndarray) -> dict[str, float]:
    s = np.asarray(samples, dtype=np.float64).ravel()
    if len(s) == 0:
        return {"max": float("nan"), "p99": float("nan"), "frac_110": float("nan"), "frac_115": float("nan"), "frac_120": float("nan")}
    return {
        "max": float(np.max(s)),
        "p99": float(np.quantile(s, 0.99)),
        "frac_110": float(np.mean(s > 110)),
        "frac_115": float(np.mean(s > 115)),
        "frac_120": float(np.mean(s > 120)),
    }


def _exceedance_compare(obs: np.ndarray, sim_flat: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    obs = np.asarray(obs, dtype=np.float64).ravel()
    sim_flat = np.asarray(sim_flat, dtype=np.float64).ravel()
    out: dict[str, Any] = {}
    for t in thresholds:
        out[f"obs_frac_gt_{t}"] = float(np.mean(obs > t)) if len(obs) else float("nan")
        out[f"sim_frac_gt_{t}"] = float(np.mean(sim_flat > t)) if len(sim_flat) else float("nan")
    return out


def _event_metrics(
    events: np.ndarray,
    ev_obs: np.ndarray,
    ev_mean_dec: np.ndarray,
    ev_mean_cal: np.ndarray,
) -> dict[str, float]:
    m = {}
    m["rmse_dec"] = float(np.sqrt(np.mean((ev_obs - ev_mean_dec) ** 2)))
    m["rmse_cal"] = float(np.sqrt(np.mean((ev_obs - ev_mean_cal) ** 2)))
    m["mae_dec"] = float(np.mean(np.abs(ev_obs - ev_mean_dec)))
    m["mae_cal"] = float(np.mean(np.abs(ev_obs - ev_mean_cal)))
    m["bias_dec"] = float(np.mean(ev_mean_dec - ev_obs))
    m["bias_cal"] = float(np.mean(ev_mean_cal - ev_obs))
    return m


def _draw_metrics_for_split(
    draws: pd.DataFrame,
    event_ids: set[int],
    ev_col: str,
    group_col: str,
    ev_obs_map: dict[int, float],
    *,
    ev_cal_col: str | None = None,
) -> dict[str, Any]:
    sub = draws[draws[group_col].astype(str).isin(event_ids)].copy()
    crps_b, crps_a = [], []
    pit_b, pit_a = [], []
    cov50_b, cov80_b, cov90_b = [], [], []
    cov50_a, cov80_a, cov90_a = [], [], []
    w50_b, w80_b, w90_b = [], [], []
    w50_a, w80_a, w90_a = [], [], []

    sim_before_all: list[float] = []
    sim_after_all: list[float] = []
    obs_list: list[float] = []

    for eid, g in sub.groupby(group_col, sort=False):
        eid_s = str(eid)
        if eid_s not in ev_obs_map:
            continue
        y = float(ev_obs_map[eid_s])
        s0 = pd.to_numeric(g[ev_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(s0) < 2:
            continue
        obs_list.append(y)
        sim_before_all.extend(s0.tolist())
        crps_b.append(_empirical_crps(s0, y))
        pit_b.append(_pit_rank(s0, y))
        for qlo, qhi, covs, widths in (
            (0.25, 0.75, cov50_b, w50_b),
            (0.10, 0.90, cov80_b, w80_b),
            (0.05, 0.95, cov90_b, w90_b),
        ):
            inside, w = _interval_stats(s0, y, qlo, qhi)
            covs.append(float(inside))
            widths.append(w)

        if ev_cal_col and ev_cal_col in g.columns:
            s1 = pd.to_numeric(g[ev_cal_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(s1) >= 2:
                sim_after_all.extend(s1.tolist())
                crps_a.append(_empirical_crps(s1, y))
                pit_a.append(_pit_rank(s1, y))
                for qlo, qhi, covs, widths in (
                    (0.25, 0.75, cov50_a, w50_a),
                    (0.10, 0.90, cov80_a, w80_a),
                    (0.05, 0.95, cov90_a, w90_a),
                ):
                    inside, w = _interval_stats(s1, y, qlo, qhi)
                    covs.append(float(inside))
                    widths.append(w)

    def _mean(xs: list[float]) -> float:
        return float(np.nanmean(xs)) if xs else float("nan")

    out: dict[str, Any] = {
        "crps_ev_mean_before": _mean(crps_b),
        "crps_ev_mean_after": _mean(crps_a) if crps_a else float("nan"),
        "pit_hist_before": np.histogram(pit_b, bins=20, range=(0, 1))[0].tolist() if pit_b else [],
        "pit_hist_after": np.histogram(pit_a, bins=20, range=(0, 1))[0].tolist() if pit_a else [],
        "coverage_50_before": _mean(cov50_b),
        "coverage_80_before": _mean(cov80_b),
        "coverage_90_before": _mean(cov90_b),
        "coverage_50_after": _mean(cov50_a) if cov50_a else float("nan"),
        "coverage_80_after": _mean(cov80_a) if cov80_a else float("nan"),
        "coverage_90_after": _mean(cov90_a) if cov90_a else float("nan"),
        "width_50_before": _mean(w50_b),
        "width_80_before": _mean(w80_b),
        "width_90_before": _mean(w90_b),
        "width_50_after": _mean(w50_a) if w50_a else float("nan"),
        "width_80_after": _mean(w80_a) if w80_a else float("nan"),
        "width_90_after": _mean(w90_a) if w90_a else float("nan"),
        "tail_before": _tail_stats(np.array(sim_before_all)),
        "tail_after": _tail_stats(np.array(sim_after_all)) if sim_after_all else {},
        "exceedance": _exceedance_compare(
            np.array(obs_list, dtype=np.float64),
            np.array(sim_before_all, dtype=np.float64),
            [100, 105, 110, 115],
        ),
    }
    if sim_after_all:
        ex2 = _exceedance_compare(
            np.array(obs_list, dtype=np.float64),
            np.array(sim_after_all, dtype=np.float64),
            [100, 105, 110, 115],
        )
        out["exceedance_after"] = ex2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Default: fit calibrator on official train+calibration events (~5644), evaluate RMSE / "
            "calibration metrics on official test (1010). Use --split-mode random for legacy 60/20/20 on one run."
        )
    )
    ap.add_argument(
        "--heldout-dir",
        type=Path,
        default=None,
        help="Single pipeline run containing predictive draws (all splits or test-only). Used when fit/eval dirs unset.",
    )
    ap.add_argument(
        "--fit-heldout-dir",
        type=Path,
        default=None,
        help="Pipeline output with draws for train+calibration events (baseline mode).",
    )
    ap.add_argument(
        "--eval-heldout-dir",
        type=Path,
        default=None,
        help="Pipeline output with draws for official test events (baseline mode). Defaults to --heldout-dir.",
    )
    ap.add_argument(
        "--split-mode",
        choices=("baseline", "random"),
        default="baseline",
        help="baseline: fit on baseline_direct_y train∪cal, metrics on test. random: 60/20/20 on one draw table.",
    )
    ap.add_argument(
        "--baseline-splits-dir",
        type=Path,
        default=SBI / "data_processed",
        help="Directory with baseline_direct_y_{train,calibration,test}.parquet",
    )
    ap.add_argument(
        "--context-master",
        type=Path,
        default=SBI / "data_processed" / "sbi_context_event_master.parquet",
        help="Parquet with event_id + context columns",
    )
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--cap", type=float, default=121.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timestamp", type=str, default=None, help="UTC folder name; default now")
    ap.add_argument(
        "--allow-nonempty-output-dir",
        action="store_true",
        help="Allow writing into non-empty output dir (disabled by default for safety).",
    )
    args = ap.parse_args()

    group_col = "event_id"
    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_root = REPO / "artifacts" / "physics_decoder_calibration" / stamp
    plots_dir = out_root / "plots"
    if out_root.exists() and any(out_root.iterdir()) and not args.allow_nonempty_output_dir:
        raise RuntimeError(f"Output dir must be new/empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _apply_mean_calibration(cal: EVPhysicsCalibrator, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        pr = cal.predict_residual(out)
        out["EV_cal_mean"] = out["EV_dec_mean"].to_numpy(dtype=np.float64) + pr
        out["pred_EV_resid"] = pr
        return out

    def _draw_tail_and_soft(df_cal: pd.DataFrame, rho: float) -> dict[str, Any]:
        ev_pre = pd.to_numeric(df_cal[ev_col], errors="coerce").to_numpy(dtype=np.float64)
        ev_post = pd.to_numeric(df_cal["EV_cal"], errors="coerce").to_numpy(dtype=np.float64)
        dm = df_cal["EV_dec_mean_merged"].to_numpy(dtype=np.float64)
        cm = df_cal["EV_cal_mean_merged"].to_numpy(dtype=np.float64)
        temp = cm + rho * (ev_pre - dm)
        n_soft = int(np.sum(ev_post < temp - 1e-4))
        return {
            "max_EV_before": float(np.nanmax(ev_pre)) if len(ev_pre) else float("nan"),
            "max_EV_after": float(np.nanmax(ev_post)) if len(ev_post) else float("nan"),
            "p99_EV_before": float(np.nanquantile(ev_pre, 0.99)) if len(ev_pre) else float("nan"),
            "p99_EV_after": float(np.nanquantile(ev_post, 0.99)) if len(ev_post) else float("nan"),
            "n_soft_capped": n_soft,
        }

    def _transform_draws_from_event_table(
        draws_in: pd.DataFrame, event_df: pd.DataFrame, *, obs_map: dict[str, float]
    ) -> pd.DataFrame:
        out = draws_in.copy()
        # Mandatory: draw transform must use exact event-level EV_cal_mean used for metrics.
        evtab = event_df[[group_col, "EV_dec_mean", "EV_cal_mean", "pred_EV_resid"]].copy()
        out = out.merge(evtab, on=group_col, how="left")
        out["EV_dec_mean_merged"] = pd.to_numeric(out["EV_dec_mean"], errors="coerce")
        out["EV_cal_mean_merged"] = pd.to_numeric(out["EV_cal_mean"], errors="coerce")
        out["EV_raw_ref"] = pd.to_numeric(out[ev_col], errors="coerce")
        out["EV_cal_raw"] = out["EV_cal_mean_merged"] + args.rho * (out["EV_raw_ref"] - out["EV_dec_mean_merged"])
        out["manual_EV_cal_raw"] = out["EV_cal_mean_merged"] + args.rho * (out["EV_raw_ref"] - out["EV_dec_mean_merged"])
        out["diff_raw"] = out["EV_cal_raw"] - out["manual_EV_cal_raw"]
        out["EV_cal"] = (
            soft_cap_ev(out["EV_cal_raw"].to_numpy(dtype=np.float64), cap=float(args.cap), tau=float(args.tau))
            if args.cap is not None
            else out["EV_cal_raw"].to_numpy(dtype=np.float64)
        )
        out["observed_EV"] = out[group_col].astype(str).map(obs_map)
        return out

    def _validate_draw_consistency(
        draws_aug: pd.DataFrame, event_df: pd.DataFrame, *, split_label: str, cap_enabled: bool, bias_tol: float = 0.25
    ) -> None:
        required = [group_col, "EV_cal", "EV_cal_raw", "manual_EV_cal_raw", "diff_raw", "EV_cal_mean_merged", "EV_dec_mean_merged"]
        missing = [c for c in required if c not in draws_aug.columns]
        if missing:
            raise RuntimeError(f"[{split_label}] Missing consistency columns: {missing}")
        raw_diff_max = float(np.nanmax(np.abs(pd.to_numeric(draws_aug["diff_raw"], errors="coerce").to_numpy(dtype=np.float64))))
        if raw_diff_max > 1e-8:
            raise RuntimeError(f"[{split_label}] raw transform mismatch max(abs(diff_raw))={raw_diff_max:.12f}")
        g = draws_aug.groupby(group_col, sort=False)
        saved_event_mean = g["EV_cal"].mean().rename("saved_EV_cal_mean")
        recalc_event_mean = g["EV_cal_raw"].mean().rename("recalc_EV_cal_raw_mean")
        event_bias_saved = float("nan")
        event_bias_event_table = float(np.mean(event_df["EV_cal_mean"].to_numpy(dtype=np.float64) - event_df["EV_obs"].to_numpy(dtype=np.float64)))
        obs_series = g["observed_EV"].first().rename("EV_obs")
        chk = pd.concat([saved_event_mean, recalc_event_mean, obs_series], axis=1).dropna()
        if len(chk):
            event_bias_saved = float(np.mean(chk["saved_EV_cal_mean"] - chk["EV_obs"]))
        merged_cmp = event_df[[group_col, "EV_cal_mean"]].copy()
        merged_cmp = merged_cmp.merge(saved_event_mean.reset_index(), on=group_col, how="inner")
        max_abs_saved_vs_recalc = float(np.nanmax(np.abs((chk["saved_EV_cal_mean"] - chk["recalc_EV_cal_raw_mean"]).to_numpy(dtype=np.float64)))) if len(chk) else float("nan")
        max_abs_saved_vs_event = float(
            np.nanmax(np.abs((merged_cmp["saved_EV_cal_mean"] - merged_cmp["EV_cal_mean"]).to_numpy(dtype=np.float64)))
        ) if len(merged_cmp) else float("nan")
        mean_cal_merged_minus_event = float(
            np.nanmean(
                pd.to_numeric(draws_aug["EV_cal_mean_merged"], errors="coerce").to_numpy(dtype=np.float64)
                - pd.to_numeric(draws_aug["EV_cal_mean"], errors="coerce").to_numpy(dtype=np.float64)
            )
        )
        max_abs_cal_merged_minus_event = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(draws_aug["EV_cal_mean_merged"], errors="coerce").to_numpy(dtype=np.float64)
                    - pd.to_numeric(draws_aug["EV_cal_mean"], errors="coerce").to_numpy(dtype=np.float64)
                )
            )
        )
        before_bias = float(np.mean(event_df["EV_dec_mean"].to_numpy(dtype=np.float64) - event_df["EV_obs"].to_numpy(dtype=np.float64)))
        after_bias = float(np.mean(event_df["EV_cal_mean"].to_numpy(dtype=np.float64) - event_df["EV_obs"].to_numpy(dtype=np.float64)))
        print(
            f"[{split_label}] consistency: "
            f"mean(EV_dec_mean-EV_obs)={before_bias:.6f}, "
            f"mean(EV_cal_mean-EV_obs)={after_bias:.6f}, "
            f"mean(saved_EV_cal_mean-EV_obs)={event_bias_saved:.6f}, "
            f"max_abs(saved_mean-recomputed_mean)={max_abs_saved_vs_recalc:.8f}, "
            f"mean(EV_cal_mean_merged-EV_cal_mean)={mean_cal_merged_minus_event:.8f}, "
            f"max_abs(EV_cal_mean_merged-EV_cal_mean)={max_abs_cal_merged_minus_event:.8f}, "
            f"max_abs(saved_event_mean-vs-event_table_mean)={max_abs_saved_vs_event:.8f}"
        )
        if abs(event_bias_saved - after_bias) > bias_tol:
            raise RuntimeError(
                f"[{split_label}] consistency failure: abs(mean(saved_EV_cal_mean-EV_obs)-mean(EV_cal_mean-EV_obs))="
                f"{abs(event_bias_saved - after_bias):.6f} > {bias_tol:.3f}"
            )
        if not cap_enabled and max_abs_saved_vs_event > 1e-8:
            raise RuntimeError(
                f"[{split_label}] cap disabled but saved draw event mean != EV_cal_mean (max diff {max_abs_saved_vs_event:.10f})"
            )

    def _qq(ax: Any, sample: np.ndarray, ref: np.ndarray, title: str) -> None:
        sample = np.sort(np.asarray(sample, dtype=np.float64))
        ref = np.sort(np.asarray(ref, dtype=np.float64))
        n = min(len(sample), len(ref))
        if n < 5:
            return
        ax.scatter(ref[:n], sample[:n], s=8, alpha=0.4)
        lo = min(ref[0], sample[0])
        hi = max(ref[-1], sample[-1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("reference quantiles")
        ax.set_ylabel("sample quantiles")

    def _make_plots(
        diag_df: pd.DataFrame,
        draws_cal: pd.DataFrame,
        obs_map: dict[str, float],
        plot_tag: str,
        metrics_for_pit: dict[str, Any],
    ) -> None:
        vo = diag_df["EV_obs"].to_numpy(dtype=np.float64)
        vd = diag_df["EV_dec_mean"].to_numpy(dtype=np.float64)
        vc = diag_df["EV_cal_mean"].to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(vd, vo, alpha=0.35, s=12, label="decoded mean")
        lims = [min(vo.min(), vd.min()) - 5, max(vo.max(), vd.max()) + 5]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("EV_dec_mean")
        ax.set_ylabel("EV_obs")
        ax.set_title(f"Observed vs decoded mean ({plot_tag})")
        ax.legend()
        fig.savefig(plots_dir / f"obs_vs_dec_mean_before__{plot_tag}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(vc, vo, alpha=0.35, s=12, color="darkgreen", label="calibrated mean")
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("EV_cal_mean")
        ax.set_ylabel("EV_obs")
        ax.set_title(f"Observed vs calibrated mean ({plot_tag})")
        ax.legend()
        fig.savefig(plots_dir / f"obs_vs_cal_mean_after__{plot_tag}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vo - vd, bins=40, alpha=0.5, label="resid before", density=True)
        ax.hist(vo - vc, bins=40, alpha=0.5, label="resid after", density=True)
        ax.legend()
        ax.set_title(f"EV residuals ({plot_tag})")
        fig.savefig(plots_dir / f"residuals_before_after__{plot_tag}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _qq(axes[0], vo, vd, f"QQ: observed vs dec mean ({plot_tag})")
        _qq(axes[1], vo, vc, f"QQ: observed vs cal mean ({plot_tag})")
        fig.tight_layout()
        fig.savefig(plots_dir / f"qq_mean_vs_obs__{plot_tag}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        pitb = metrics_for_pit.get("pit_hist_before", [])
        pita = metrics_for_pit.get("pit_hist_after", [])
        if pitb:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(np.linspace(0, 1, len(pitb), endpoint=False), pitb, width=0.04)
            axes[0].set_title(f"PIT before ({plot_tag} draws)")
            if pita:
                axes[1].bar(np.linspace(0, 1, len(pita), endpoint=False), pita, width=0.04)
                axes[1].set_title(f"PIT after ({plot_tag} draws)")
            fig.savefig(plots_dir / f"pit_hist_before_after__{plot_tag}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        if len(draws_cal):
            eid_pick = draws_cal.groupby(group_col).size().idxmax()
            g0 = draws_cal[draws_cal[group_col] == eid_pick]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(pd.to_numeric(g0[ev_col], errors="coerce").dropna(), bins=40, alpha=0.5, density=True, label="EV_dec draws")
            ax.hist(pd.to_numeric(g0["EV_cal"], errors="coerce").dropna(), bins=40, alpha=0.5, density=True, label="EV_cal draws")
            y0 = float(obs_map.get(str(eid_pick), float("nan")))
            if np.isfinite(y0):
                ax.axvline(y0, color="red", lw=2, label="EV_obs")
            ax.legend()
            ax.set_title(f"EV draw overlay ({plot_tag} event_id={eid_pick})")
            fig.savefig(plots_dir / f"ev_draw_overlay_example__{plot_tag}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    # --- baseline split (recommended) ---
    if args.split_mode == "baseline":
        _, _, baseline_test_ids, train_cal_ids = _load_baseline_split_ids(Path(args.baseline_splits_dir))
        obs_by_event = _observed_ev_from_baseline(Path(args.baseline_splits_dir))

        fit_dir = Path(args.fit_heldout_dir).resolve() if args.fit_heldout_dir else None
        eval_dir = Path(args.eval_heldout_dir).resolve() if args.eval_heldout_dir else None
        single_dir = Path(args.heldout_dir).resolve() if args.heldout_dir else None

        if fit_dir and eval_dir:
            draws_fit, ev_col, draws_path_fit = _load_draws_from_run(fit_dir, group_col)
            draws_eval, ev_col_e, draws_path_eval = _load_draws_from_run(eval_dir, group_col)
            if ev_col_e != ev_col:
                raise ValueError(f"Decoder EV column mismatch: fit {ev_col!r} vs eval {ev_col_e!r}")
            draws_fit = draws_fit[draws_fit[group_col].astype(str).isin(train_cal_ids)].copy()
            draws_eval = draws_eval[draws_eval[group_col].astype(str).isin(baseline_test_ids)].copy()
            draws_path = draws_path_fit
        elif single_dir:
            if not single_dir.is_dir():
                raise FileNotFoundError(single_dir)
            draws_all, ev_col, draws_path = _load_draws_from_run(single_dir, group_col)
            draws_fit = draws_all[draws_all[group_col].astype(str).isin(train_cal_ids)].copy()
            draws_eval = draws_all[draws_all[group_col].astype(str).isin(baseline_test_ids)].copy()
            draws_path_fit = draws_path_eval = draws_path
        else:
            raise ValueError(
                "baseline mode needs either (--fit-heldout-dir and --eval-heldout-dir) or a single --heldout-dir "
                "whose draws include both train∪calibration and test event_ids."
            )

        summ_fit = _build_event_summary_table(
            draws_fit, ev_col, group_col, obs_by_event, Path(args.context_master)
        )
        summ_eval = _build_event_summary_table(
            draws_eval, ev_col, group_col, obs_by_event, Path(args.context_master)
        )
        _assert_residual_convention(summ_fit, label="baseline_fit")
        _assert_residual_convention(summ_eval, label="baseline_eval")
        if summ_fit.shape[0] < 100:
            raise RuntimeError(
                f"Too few train+calibration events with draws+obs ({summ_fit.shape[0]}). "
                "Run the held-out pipeline on train∪calibration event_ids (or pass --fit-heldout-dir)."
            )
        if summ_eval.shape[0] < 10:
            raise RuntimeError(
                f"Too few official test events with draws+obs ({summ_eval.shape[0]}). "
                "Pass test-run draws via --eval-heldout-dir or --heldout-dir."
            )

        got_fit_ids = set(summ_fit[group_col].astype(str))
        missing_fit = train_cal_ids - got_fit_ids
        got_eval_ids = set(summ_eval[group_col].astype(str))
        missing_test = baseline_test_ids - got_eval_ids
        if missing_fit:
            print(
                f"WARNING: {len(missing_fit)} train∪calibration event_ids lack draws or EV_obs in this run "
                f"(showing up to 5): {sorted(missing_fit)[:5]}",
                file=sys.stderr,
            )
        if missing_test:
            print(
                f"WARNING: {len(missing_test)} official test event_ids missing from eval draws "
                f"(showing up to 5): {sorted(missing_test)[:5]}",
                file=sys.stderr,
            )

        numeric_features, categorical_features = _resolve_feature_lists(summ_fit)
        numeric_features = [c for c in numeric_features if c in summ_fit.columns and c in summ_eval.columns]
        categorical_features = [c for c in categorical_features if c in summ_fit.columns and c in summ_eval.columns]

        cal = EVPhysicsCalibrator(
            rho=args.rho,
            cap=args.cap,
            tau=args.tau,
            ev_col=ev_col,
            group_col=group_col,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        train_cal_df = summ_fit.dropna(subset=["EV_resid"]).copy()
        _log_pre_fit_residual_stats(train_cal_df, label="baseline_train_cal")
        cal.fit(train_cal_df)

        train_cal_calib = _apply_mean_calibration(cal, summ_fit)
        test_df = _apply_mean_calibration(cal, summ_eval)
        _log_post_fit_residual_diagnostics(train_cal_calib, label="baseline_train_cal")
        _log_post_fit_residual_diagnostics(test_df, label="baseline_test")

        metrics_train_cal = _event_metrics(
            train_cal_calib[group_col].to_numpy(),
            train_cal_calib["EV_obs"].to_numpy(dtype=np.float64),
            train_cal_calib["EV_dec_mean"].to_numpy(dtype=np.float64),
            train_cal_calib["EV_cal_mean"].to_numpy(dtype=np.float64),
        )
        metrics_test = _event_metrics(
            test_df[group_col].to_numpy(),
            test_df["EV_obs"].to_numpy(dtype=np.float64),
            test_df["EV_dec_mean"].to_numpy(dtype=np.float64),
            test_df["EV_cal_mean"].to_numpy(dtype=np.float64),
        )

        ev_obs_map_full = {**obs_by_event}

        draws_train_cal_cal = _transform_draws_from_event_table(draws_fit, train_cal_calib, obs_map=obs_by_event)
        draws_test_cal = _transform_draws_from_event_table(draws_eval, test_df, obs_map=obs_by_event)
        _validate_draw_consistency(draws_train_cal_cal, train_cal_calib, split_label="baseline_train_cal", cap_enabled=(args.cap is not None))
        _validate_draw_consistency(draws_test_cal, test_df, split_label="baseline_test", cap_enabled=(args.cap is not None))

        train_cal_id_str = {str(x) for x in summ_fit[group_col].astype(str)}
        test_id_str = {str(x) for x in summ_eval[group_col].astype(str)}

        metrics_train_cal["draw_tail"] = _draw_tail_and_soft(draws_train_cal_cal, args.rho)
        metrics_test["draw_tail"] = _draw_tail_and_soft(draws_test_cal, args.rho)
        metrics_train_cal["distributional"] = _draw_metrics_for_split(
            draws_train_cal_cal, train_cal_id_str, ev_col, group_col, ev_obs_map_full, ev_cal_col="EV_cal"
        )
        metrics_test["distributional"] = _draw_metrics_for_split(
            draws_test_cal, test_id_str, ev_col, group_col, ev_obs_map_full, ev_cal_col="EV_cal"
        )

        cal.save(out_root / "ev_calibrator.joblib")
        summ_out = pd.concat(
            [
                train_cal_calib.assign(split_partition="train_or_calibration"),
                test_df.assign(split_partition="official_test"),
            ],
            ignore_index=True,
        )
        summ_out.to_csv(out_root / "event_level_calibration_table.csv", index=False)
        draws_train_cal_cal.to_parquet(out_root / "calibrated_draws_train_calibration.parquet", index=False)
        draws_test_cal.to_parquet(out_root / "calibrated_draws_test.parquet", index=False)
        dbg = draws_test_cal.sample(n=min(100, len(draws_test_cal)), random_state=args.seed).copy()
        dbg_cols = [
            group_col,
            "draw_idx",
            ev_col,
            "EV_dec_mean_merged",
            "pred_EV_resid",
            "EV_cal_mean_merged",
            "EV_cal_raw",
            "EV_cal",
            "observed_EV",
            "manual_EV_cal_raw",
            "diff_raw",
        ]
        dbg_cols = [c for c in dbg_cols if c in dbg.columns]
        dbg[dbg_cols].to_csv(out_root / "draw_transform_debug_sample.csv", index=False)
        (out_root / "metrics_train_calibration.json").write_text(json.dumps(metrics_train_cal, indent=2), encoding="utf-8")
        (out_root / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")
        (out_root / "feature_manifest.json").write_text(json.dumps(cal.feature_manifest(), indent=2), encoding="utf-8")

        cfg = {
            "split_mode": "baseline",
            "baseline_splits_dir": str(Path(args.baseline_splits_dir).resolve()),
            "fit_heldout_dir": str(fit_dir) if fit_dir else None,
            "eval_heldout_dir": str(eval_dir) if eval_dir else None,
            "single_heldout_dir": str(single_dir) if single_dir else None,
            "draws_file_fit": str(draws_path_fit),
            "draws_file_eval": str(draws_path_eval) if fit_dir and eval_dir else str(draws_path),
            "n_fit_events_summary": int(summ_fit.shape[0]),
            "n_test_events_summary": int(summ_eval.shape[0]),
            "n_baseline_train_cal": len(train_cal_ids),
            "n_baseline_test": len(baseline_test_ids),
            "rho": args.rho,
            "cap": args.cap,
            "tau": args.tau,
            "seed": args.seed,
            "ev_col_resolved": ev_col,
        }
        try:
            import yaml

            (out_root / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except ImportError:
            (out_root / "config_resolved.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        _make_plots(test_df, draws_test_cal, obs_by_event, "official_test", metrics_test["distributional"])
        _make_plots(train_cal_calib, draws_train_cal_cal, obs_by_event, "train_calibration", metrics_train_cal["distributional"])

        hash_targets = [
            out_root / "ev_calibrator.joblib",
            out_root / "calibrated_draws_test.parquet",
            out_root / "calibrated_draws_train_calibration.parquet",
            out_root / "event_level_calibration_table.csv",
            out_root / "metrics_test.json",
        ]
        print("Output file SHA256:")
        for hp in hash_targets:
            print(f"  {hp.name}: {_sha256(hp)}")
        prev_dirs = sorted(
            [p for p in (REPO / "artifacts" / "physics_decoder_calibration").glob("*") if p.is_dir() and p.name != stamp]
        )
        if prev_dirs:
            prev = prev_dirs[-1]
            identical = []
            for hp in hash_targets:
                q = prev / hp.name
                if q.exists() and _sha256(q) == _sha256(hp):
                    identical.append(hp.name)
            if identical:
                print(
                    f"WARNING: identical output hashes vs previous run {prev.name}: {identical}. "
                    "If code/config changed, this may indicate stale or unchanged outputs.",
                    file=sys.stderr,
                )

        print(f"Wrote calibration bundle to {out_root}")
        print(
            json.dumps(
                {
                    "train_calibration_headline": {k: metrics_train_cal[k] for k in metrics_train_cal if k != "distributional"},
                    "official_test_headline": {k: metrics_test[k] for k in metrics_test if k != "distributional"},
                },
                indent=2,
            )
        )
        return

    # --- random 60/20/20 (legacy): one heldout dir ---
    if args.heldout_dir is None:
        raise ValueError("--heldout-dir is required for --split-mode random")
    heldout = Path(args.heldout_dir).resolve()
    if not heldout.is_dir():
        raise FileNotFoundError(heldout)

    draws_path = _find_draws_file(heldout)
    draws = _read_table(draws_path)
    ev_col = _resolve_col(draws, ["EV", "EV_dec", "ev_dec", "ev_mph"], "decoder EV")
    _ensure_event_id_column(draws, group_col)

    obs_by_event: dict[str, float] = {}
    obs_by_event = _merge_run_summary_obs(heldout, group_col, obs_by_event)
    obs_by_event = _merge_draws_column_obs(draws, group_col, obs_by_event)
    if not obs_by_event:
        raise ValueError("Need observed EV on draws or predictive_summary_by_event.*")

    summ = _build_event_summary_table(draws, ev_col, group_col, obs_by_event, Path(args.context_master))
    _assert_residual_convention(summ, label="random_all")
    numeric_features, categorical_features = _resolve_feature_lists(summ)

    rng = np.random.default_rng(args.seed)
    u_ids = summ[group_col].astype(str).unique()
    rng.shuffle(u_ids)
    n = len(u_ids)
    i1, i2 = int(0.6 * n), int(0.8 * n)
    train_ids = set(u_ids[:i1].tolist())
    val_ids = set(u_ids[i1:i2].tolist())
    test_ids = set(u_ids[i2:].tolist())

    train_df = summ[summ[group_col].astype(str).isin(train_ids)].copy()
    val_df = summ[summ[group_col].astype(str).isin(val_ids)].copy()
    test_df = summ[summ[group_col].astype(str).isin(test_ids)].copy()

    numeric_features = [c for c in numeric_features if c in train_df.columns]
    categorical_features = [c for c in categorical_features if c in train_df.columns]

    cal = EVPhysicsCalibrator(
        rho=args.rho,
        cap=args.cap,
        tau=args.tau,
        ev_col=ev_col,
        group_col=group_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    train_df = train_df.dropna(subset=["EV_resid"])
    _assert_residual_convention(train_df, label="random_train")
    _assert_residual_convention(val_df, label="random_val")
    _assert_residual_convention(test_df, label="random_test")
    _log_pre_fit_residual_stats(train_df, label="random_train")
    cal.fit(train_df)

    val_df = _apply_mean_calibration(cal, val_df)
    test_df = _apply_mean_calibration(cal, test_df)
    _log_post_fit_residual_diagnostics(val_df, label="random_val")
    _log_post_fit_residual_diagnostics(test_df, label="random_test")

    metrics_val = _event_metrics(
        val_df[group_col].to_numpy(),
        val_df["EV_obs"].to_numpy(dtype=np.float64),
        val_df["EV_dec_mean"].to_numpy(dtype=np.float64),
        val_df["EV_cal_mean"].to_numpy(dtype=np.float64),
    )
    metrics_test = _event_metrics(
        test_df[group_col].to_numpy(),
        test_df["EV_obs"].to_numpy(dtype=np.float64),
        test_df["EV_dec_mean"].to_numpy(dtype=np.float64),
        test_df["EV_cal_mean"].to_numpy(dtype=np.float64),
    )

    ev_obs_map_full = {str(k): float(v) for k, v in zip(summ[group_col].to_numpy(), summ["EV_obs"].to_numpy()) if np.isfinite(v)}

    draws_val = draws[draws[group_col].astype(str).isin(val_ids)].copy()
    draws_test = draws[draws[group_col].astype(str).isin(test_ids)].copy()

    draws_val_cal = _transform_draws_from_event_table(draws_val, val_df, obs_map=obs_by_event)
    draws_test_cal = _transform_draws_from_event_table(draws_test, test_df, obs_map=obs_by_event)
    _validate_draw_consistency(draws_val_cal, val_df, split_label="random_val", cap_enabled=(args.cap is not None))
    _validate_draw_consistency(draws_test_cal, test_df, split_label="random_test", cap_enabled=(args.cap is not None))

    val_id_str = {str(x) for x in val_ids}
    test_id_str = {str(x) for x in test_ids}
    metrics_val["draw_tail"] = _draw_tail_and_soft(draws_val_cal, args.rho)
    metrics_test["draw_tail"] = _draw_tail_and_soft(draws_test_cal, args.rho)

    metrics_val["distributional"] = _draw_metrics_for_split(
        draws_val_cal, val_id_str, ev_col, group_col, ev_obs_map_full, ev_cal_col="EV_cal"
    )
    metrics_test["distributional"] = _draw_metrics_for_split(
        draws_test_cal, test_id_str, ev_col, group_col, ev_obs_map_full, ev_cal_col="EV_cal"
    )

    cal.save(out_root / "ev_calibrator.joblib")
    summ_out = pd.concat(
        [train_df.assign(split_partition="random_train"), val_df.assign(split_partition="random_val"), test_df.assign(split_partition="random_test")],
        ignore_index=True,
    )
    summ_out.to_csv(out_root / "event_level_calibration_table.csv", index=False)

    draws_val_cal.to_parquet(out_root / "calibrated_draws_validation.parquet", index=False)
    draws_test_cal.to_parquet(out_root / "calibrated_draws_test.parquet", index=False)
    dbg = draws_test_cal.sample(n=min(100, len(draws_test_cal)), random_state=args.seed).copy()
    dbg_cols = [
        group_col,
        "draw_idx",
        ev_col,
        "EV_dec_mean_merged",
        "pred_EV_resid",
        "EV_cal_mean_merged",
        "EV_cal_raw",
        "EV_cal",
        "observed_EV",
        "manual_EV_cal_raw",
        "diff_raw",
    ]
    dbg_cols = [c for c in dbg_cols if c in dbg.columns]
    dbg[dbg_cols].to_csv(out_root / "draw_transform_debug_sample.csv", index=False)

    (out_root / "metrics_validation.json").write_text(json.dumps(metrics_val, indent=2), encoding="utf-8")
    (out_root / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2), encoding="utf-8")
    (out_root / "feature_manifest.json").write_text(json.dumps(cal.feature_manifest(), indent=2), encoding="utf-8")

    cfg = {
        "split_mode": "random",
        "heldout_dir": str(heldout),
        "draws_file": str(draws_path),
        "rho": args.rho,
        "cap": args.cap,
        "tau": args.tau,
        "seed": args.seed,
        "n_train_events": len(train_ids),
        "n_val_events": len(val_ids),
        "n_test_events": len(test_ids),
        "ev_col_resolved": ev_col,
    }
    try:
        import yaml

        (out_root / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except ImportError:
        (out_root / "config_resolved.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    _make_plots(val_df, draws_val_cal, obs_by_event, "random_val", metrics_val["distributional"])

    hash_targets = [
        out_root / "ev_calibrator.joblib",
        out_root / "calibrated_draws_test.parquet",
        out_root / "calibrated_draws_validation.parquet",
        out_root / "event_level_calibration_table.csv",
        out_root / "metrics_test.json",
    ]
    print("Output file SHA256:")
    for hp in hash_targets:
        print(f"  {hp.name}: {_sha256(hp)}")
    prev_dirs = sorted(
        [p for p in (REPO / "artifacts" / "physics_decoder_calibration").glob("*") if p.is_dir() and p.name != stamp]
    )
    if prev_dirs:
        prev = prev_dirs[-1]
        identical = []
        for hp in hash_targets:
            q = prev / hp.name
            if q.exists() and _sha256(q) == _sha256(hp):
                identical.append(hp.name)
        if identical:
            print(
                f"WARNING: identical output hashes vs previous run {prev.name}: {identical}. "
                "If code/config changed, this may indicate stale or unchanged outputs.",
                file=sys.stderr,
            )

    print(f"Wrote calibration bundle to {out_root}")
    print(json.dumps({"validation": metrics_val, "test_headline": {k: metrics_test[k] for k in metrics_test if k != "distributional"}}, indent=2))


if __name__ == "__main__":
    main()
