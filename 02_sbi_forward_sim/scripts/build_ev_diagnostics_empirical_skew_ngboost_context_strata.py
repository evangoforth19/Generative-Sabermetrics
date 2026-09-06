#!/usr/bin/env python3
"""
Stratify the EV diagnostics table (empirical admissible draws, skew-normal refit, NGBoost) by
**global** training+calibration support for the Statcast context
`(pitch_type, location_bin, z_count)`.

- **location_bin:** joint label from **tertile cuts** of `plate_x` and `plate_z` fit on
  `baseline_direct_y_train` + `baseline_direct_y_calibration` only (same 3×3 zone idea as
  `build_black_box_baseline_comparison` slice axes, but as a single categorical key).
- **Support count:** number of rows in train+cal with the same `(pitch_type, location_bin, z_count)`.

Writes `ev_diagnostics_empirical_skew_ngboost_context_strata.json` and `.md` under --heldout-run.

Example:
  python scripts/build_ev_diagnostics_empirical_skew_ngboost_context_strata.py \\
    --heldout-run outputs/heldout_pipeline_test/20260409_182635Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
_NGB_DEFAULT = _ROOT / "outputs/black_box_baselines/20260408_044530Z/per_event_predictive_scores_ngboost_style.csv"

CONTEXT_KEYS = ("pitch_type", "location_bin", "z_count")


def _tertile_bin_edges(s: pd.Series, q: int = 3) -> np.ndarray:
    _, bins = pd.qcut(s.astype(np.float64), q=q, retbins=True, duplicates="drop")
    return np.asarray(bins, dtype=np.float64)


def add_location_bin_from_ref(ref: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    bx = _tertile_bin_edges(ref["plate_x"])
    bz = _tertile_bin_edges(ref["plate_z"])
    cx = pd.cut(df["plate_x"].astype(np.float64), bins=bx, include_lowest=True)
    cz = pd.cut(df["plate_z"].astype(np.float64), bins=bz, include_lowest=True)
    return cx.astype(str) + "|" + cz.astype(str)


def coverage_interval_normal(
    y: np.ndarray, mu: np.ndarray, sig: np.ndarray, q_lo: float, q_hi: float
) -> np.ndarray:
    sig = np.maximum(sig.astype(np.float64), 1e-8)
    lo = stats.norm.ppf(q_lo, loc=mu, scale=sig)
    hi = stats.norm.ppf(q_hi, loc=mu, scale=sig)
    return ((lo <= y) & (y <= hi)).astype(np.float64)


def agg_ngboost_ev(df: pd.DataFrame) -> dict[str, float | int | None]:
    n = int(len(df))
    if n == 0:
        return {
            "n_events": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "NLL": float("nan"),
            "COV50": float("nan"),
            "COV80": float("nan"),
            "COV90": float("nan"),
            "CRPS": float("nan"),
        }
    y = df["obs_EV"].to_numpy(dtype=np.float64)
    mu = df["pred_mean_EV"].to_numpy(dtype=np.float64)
    sig = np.maximum(df["pred_std_EV"].to_numpy(dtype=np.float64), 1e-8)
    return {
        "n_events": n,
        "MAE": float(np.mean(np.abs(mu - y))),
        "RMSE": float(np.sqrt(np.mean((mu - y) ** 2))),
        "NLL": float(np.nanmean(df["nll_EV"].to_numpy())),
        "COV50": float(coverage_interval_normal(y, mu, sig, 0.25, 0.75).mean()),
        "COV80": float(coverage_interval_normal(y, mu, sig, 0.10, 0.90).mean()),
        "COV90": float(coverage_interval_normal(y, mu, sig, 0.05, 0.95).mean()),
        "CRPS": float(np.nanmean(df["crps_EV"].to_numpy())),
    }


def _empirical_crps(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n < 2:
        return float("nan")
    e1 = np.mean(np.abs(s - y))
    e2 = np.mean(np.abs(s.reshape(-1, 1) - s.reshape(1, -1)))
    return float(e1 - 0.5 * e2)


def _coverage_flag(samples: np.ndarray, y: float, qlo: float, qhi: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    if len(s) < 2:
        return float("nan")
    lo, hi = np.quantile(s, [qlo, qhi])
    return float(bool(lo <= y <= hi))


def _load_summary(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "predictive_summary_by_event.parquet"
    csv = run_dir / "predictive_summary_by_event.csv"
    if pq.is_file():
        return pd.read_parquet(pq)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No predictive_summary in {run_dir}")


def _load_draws_ev(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "predictive_draws_admissible.parquet"
    csv = run_dir / "predictive_draws_admissible.csv"
    if pq.is_file():
        return pd.read_parquet(pq, columns=["event_id", "EV"])
    if csv.is_file():
        return pd.read_csv(csv, usecols=["event_id", "EV"])
    raise FileNotFoundError(f"No predictive_draws_admissible in {run_dir}")


def agg_empirical_ev(df_ev: pd.DataFrame, draw_by_eid: dict[int, np.ndarray], eids: set[int]) -> dict[str, float | int | None]:
    mae_l: list[float] = []
    rmse_l: list[float] = []
    crps_l: list[float] = []
    c50: list[float] = []
    c80: list[float] = []
    c90: list[float] = []
    sub = df_ev.loc[df_ev["event_id"].isin(eids)]
    if len(sub) == 0:
        return {
            "n_events": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "NLL": None,
            "COV50": float("nan"),
            "COV80": float("nan"),
            "COV90": float("nan"),
            "CRPS": float("nan"),
        }
    for _, er in sub.iterrows():
        eid = int(er["event_id"])
        obs = float(er["observed_EV"])
        pred_mean = float(er["pred_EV_mean"])
        if not np.isfinite(pred_mean):
            pred_mean = float("nan")
        err = pred_mean - obs
        mae_l.append(abs(err))
        rmse_l.append(err**2)
        samps = draw_by_eid.get(eid, np.empty(0, dtype=np.float64))
        if len(samps) > 0:
            crps_l.append(_empirical_crps(samps, obs))
            c50.append(_coverage_flag(samps, obs, 0.25, 0.75))
            c80.append(_coverage_flag(samps, obs, 0.10, 0.90))
            c90.append(_coverage_flag(samps, obs, 0.05, 0.95))
        else:
            crps_l.append(float("nan"))
            c50.append(float("nan"))
            c80.append(float("nan"))
            c90.append(float("nan"))
    return {
        "n_events": int(len(sub)),
        "MAE": float(np.nanmean(mae_l)),
        "RMSE": float(np.sqrt(np.nanmean(rmse_l))),
        "NLL": None,
        "COV50": float(np.nanmean(c50)),
        "COV80": float(np.nanmean(c80)),
        "COV90": float(np.nanmean(c90)),
        "CRPS": float(np.nanmean(crps_l)),
    }


def agg_skew_ev(df_sk: pd.DataFrame, eids: set[int]) -> dict[str, float | int]:
    sub = df_sk.loc[df_sk["event_id"].isin(eids)]
    n = int(len(sub))
    if n == 0:
        return {
            "n_events": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "NLL": float("nan"),
            "COV50": float("nan"),
            "COV80": float("nan"),
            "COV90": float("nan"),
            "CRPS": float("nan"),
        }
    return {
        "n_events": n,
        "MAE": float(sub["abs_error"].mean()),
        "RMSE": float(np.sqrt(sub["sq_error"].mean())),
        "NLL": float(sub["nll"].mean()),
        "COV50": float(sub["cov50"].mean()),
        "COV80": float(sub["cov80"].mean()),
        "COV90": float(sub["cov90"].mean()),
        "CRPS": float(sub["crps"].mean()),
    }


def row_metrics(d: dict) -> dict:
    keys = ["MAE", "RMSE", "NLL", "COV50", "COV80", "COV90", "CRPS"]
    out = {k: d[k] for k in keys}
    out["n_events"] = d["n_events"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-run", type=Path, required=True)
    ap.add_argument("--ngboost-scores", type=Path, default=_NGB_DEFAULT)
    ap.add_argument(
        "--baseline-train",
        type=Path,
        default=_ROOT / "data_processed/baseline_direct_y_train.parquet",
    )
    ap.add_argument(
        "--baseline-cal",
        type=Path,
        default=_ROOT / "data_processed/baseline_direct_y_calibration.parquet",
    )
    ap.add_argument(
        "--baseline-test",
        type=Path,
        default=_ROOT / "data_processed/baseline_direct_y_test.parquet",
    )
    args = ap.parse_args()
    run_dir = args.heldout_run.resolve()
    train_p = args.baseline_train.resolve()
    cal_p = args.baseline_cal.resolve()
    test_p = args.baseline_test.resolve()
    for p in (train_p, cal_p, test_p):
        if not p.is_file():
            raise FileNotFoundError(p)

    need_cols = ["event_id", "pitch_type", "plate_x", "plate_z", "z_count"]
    train = pd.read_parquet(train_p, columns=need_cols)
    cal = pd.read_parquet(cal_p, columns=need_cols)
    test = pd.read_parquet(test_p, columns=need_cols)
    test = test.drop_duplicates("event_id", keep="first")

    ref = pd.concat([train, cal], ignore_index=True)
    ref = ref.assign(location_bin=add_location_bin_from_ref(ref, ref))
    test = test.assign(location_bin=add_location_bin_from_ref(ref, test))

    hist = ref[list(CONTEXT_KEYS)]
    support = hist.groupby(list(CONTEXT_KEYS), dropna=False).size().rename("n_context_train_cal").reset_index()

    meta = test.merge(support, on=list(CONTEXT_KEYS), how="left")
    meta["n_context_train_cal"] = meta["n_context_train_cal"].fillna(0).astype(int)

    sk_path = run_dir / "ev_skewnormal_refit_per_event.csv"
    if not sk_path.is_file():
        raise FileNotFoundError(f"Missing {sk_path.name}; run skew-normal refit evaluation for {run_dir}")
    ng_path = args.ngboost_scores.resolve()
    if not ng_path.is_file():
        raise FileNotFoundError(ng_path)

    summ_pq = run_dir / "predictive_summary_by_event.parquet"
    summ_csv = run_dir / "predictive_summary_by_event.csv"
    held_ids: set[int]
    if summ_pq.is_file():
        held_ids = set(pd.read_parquet(summ_pq, columns=["event_id"])["event_id"].astype(int))
    elif summ_csv.is_file():
        held_ids = set(pd.read_csv(summ_csv, usecols=["event_id"])["event_id"].astype(int))
    else:
        held_ids = set(pd.read_csv(ng_path)["event_id"].astype(int))

    meta = meta.loc[meta["event_id"].isin(held_ids)].copy()
    if meta["event_id"].nunique() != len(held_ids):
        raise ValueError(
            f"baseline test events {meta['event_id'].nunique()} != held-out summary {len(held_ids)}"
        )

    df_ev = _load_summary(run_dir)
    df_draw = _load_draws_ev(run_dir)
    draw_by_eid = {int(eid): g["EV"].to_numpy(dtype=np.float64) for eid, g in df_draw.groupby("event_id")}

    df_sk = pd.read_csv(sk_path)
    ng_df = pd.read_csv(ng_path)
    ng_df = ng_df.loc[ng_df["event_id"].isin(held_ids)].copy()
    if len(ng_df) != len(held_ids):
        raise ValueError(f"NGBoost rows {len(ng_df)} != held-out events {len(held_ids)}")

    strata_defs = [
        ("n_context_train_cal_lt_10", meta["n_context_train_cal"] < 10, "fewer than 10 training+calibration rows"),
        ("n_context_train_cal_eq_0", meta["n_context_train_cal"] == 0, "0 training+calibration rows"),
        ("n_context_train_cal_lt_3", meta["n_context_train_cal"] < 3, "fewer than 3 training+calibration rows"),
    ]

    keys = ["MAE", "RMSE", "NLL", "COV50", "COV80", "COV90", "CRPS"]
    strata_out: dict = {}

    for key, mask, human in strata_defs:
        eids = set(meta.loc[mask, "event_id"].astype(int).tolist())
        strata_out[key] = {
            "description": human,
            "n_events": len(eids),
            "empirical_admissible_draws_EV": row_metrics(agg_empirical_ev(df_ev, draw_by_eid, eids)),
            "skew_normal_refit_EV": row_metrics(agg_skew_ev(df_sk, eids)),
            "ngboost_gaussian_EV": row_metrics(agg_ngboost_ev(ng_df.loc[ng_df["event_id"].isin(eids)])),
        }

    out = {
        "heldout_run": str(run_dir),
        "context_keys": list(CONTEXT_KEYS),
        "location_bin_method": (
            "Joint label plate_x_tertile|plate_z_tertile: tertile bin edges from "
            "train+cal only (pd.qcut q=3, duplicates='drop'); held-out rows binned with pd.cut."
        ),
        "support_count_source": "Rows in baseline_direct_y_train + baseline_direct_y_calibration.",
        "ngboost_scores_file": str(ng_path),
        "strata": strata_out,
    }

    json_path = run_dir / "ev_diagnostics_empirical_skew_ngboost_context_strata.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def fmt(x: float | None) -> str:
        if x is None:
            return "—"
        xf = float(x)
        if not np.isfinite(xf):
            return "—"
        return f"{xf:.6f}"

    lines = [
        "# EV diagnostics by context support: empirical vs skew-normal vs NGBoost",
        "",
        f"- **Held-out run:** `{run_dir.name}`",
        f"- **Context:** `(pitch_type, location_bin, z_count)` with **location_bin** = joint tertile zones of Statcast `plate_x` and `plate_z` (edges fit on train+cal only).",
        f"- **Support count:** rows in train+cal sharing that context.",
        f"- **NGBoost file:** `{ng_path.relative_to(_ROOT)}`",
        "",
        "Metrics are **nan-mean over held-out events** in each subset (same definitions as `ev_diagnostics_empirical_skew_ngboost.md`).",
        "",
    ]

    for key, _, human in strata_defs:
        block = strata_out[key]
        lines.extend(
            [
                f"## {human}",
                "",
                f"- **Events in subset:** {block['n_events']}",
                "",
                "| Metric | Empirical (admissible draws) | Skew-normal refit | NGBoost (EV) |",
                "|--------|-----------------------------:|------------------:|-------------:|",
            ]
        )
        emp = {k: block["empirical_admissible_draws_EV"][k] for k in keys}
        skw = {k: block["skew_normal_refit_EV"][k] for k in keys}
        ngb = {k: block["ngboost_gaussian_EV"][k] for k in keys}
        for k in keys:
            lines.append(f"| {k} | {fmt(emp[k])} | {fmt(skw[k])} | {fmt(ngb[k])} |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- **Empirical:** sample mean for MAE/RMSE; sample quantiles for coverage; empirical CRPS; NLL omitted.",
            "- **Skew-normal:** per-event fit on admissible EV draws (same as main EV diagnostics report).",
            "- **NGBoost:** Gaussian EV marginal from baseline scores CSV.",
            "",
            "## JSON",
            "",
            f"- `{json_path.name}`",
            "",
        ]
    )

    md_path = run_dir / "ev_diagnostics_empirical_skew_ngboost_context_strata.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True, "md": str(md_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
