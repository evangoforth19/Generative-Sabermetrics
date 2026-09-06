#!/usr/bin/env python3
"""
Single EV metric table for a held-out run: **empirical** admissible draws, **skew-normal** refit,
and **NGBoost** (context-only Gaussian; same held-out event_ids).

Metrics: MAE, RMSE, NLL, COV50, COV80, COV90, CRPS (nan-mean over events).

Prerequisites under --heldout-run:
  - empirical_admissible_draw_benchmark.json (from build_empirical_admissible_draw_benchmark_report.py)
  - ev_skewnormal_refit_summary.json (from build_ev_skewnormal_refit_evaluation.py)

Example:
  python scripts/build_ev_diagnostics_empirical_skew_ngboost_report.py \\
    --heldout-run outputs/heldout_pipeline_test/20260409_STAMPZ
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
_NGB_DEFAULT = _ROOT / "outputs/black_box_baselines/20260408_044530Z/per_event_predictive_scores_ngboost_style.csv"


def coverage_interval_normal(
    y: np.ndarray, mu: np.ndarray, sig: np.ndarray, q_lo: float, q_hi: float
) -> np.ndarray:
    sig = np.maximum(sig.astype(np.float64), 1e-8)
    lo = stats.norm.ppf(q_lo, loc=mu, scale=sig)
    hi = stats.norm.ppf(q_hi, loc=mu, scale=sig)
    return ((lo <= y) & (y <= hi)).astype(np.float64)


def agg_ngboost_ev(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["obs_EV"].to_numpy(dtype=np.float64)
    mu = df["pred_mean_EV"].to_numpy(dtype=np.float64)
    sig = np.maximum(df["pred_std_EV"].to_numpy(dtype=np.float64), 1e-8)
    return {
        "n_events": int(len(df)),
        "MAE": float(np.mean(np.abs(mu - y))),
        "RMSE": float(np.sqrt(np.mean((mu - y) ** 2))),
        "NLL": float(np.nanmean(df["nll_EV"].to_numpy())),
        "COV50": float(coverage_interval_normal(y, mu, sig, 0.25, 0.75).mean()),
        "COV80": float(coverage_interval_normal(y, mu, sig, 0.10, 0.90).mean()),
        "COV90": float(coverage_interval_normal(y, mu, sig, 0.05, 0.95).mean()),
        "CRPS": float(np.nanmean(df["crps_EV"].to_numpy())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-run", type=Path, required=True)
    ap.add_argument(
        "--ngboost-scores",
        type=Path,
        default=_NGB_DEFAULT,
        help="per_event_predictive_scores_ngboost_style.csv",
    )
    ap.add_argument(
        "--ensure-empirical-json",
        action="store_true",
        help="If empirical_admissible_draw_benchmark.json is missing, run build_empirical_admissible_draw_benchmark_report.py first.",
    )
    args = ap.parse_args()
    run_dir = args.heldout_run.resolve()
    emp_path = run_dir / "empirical_admissible_draw_benchmark.json"
    sk_path = run_dir / "ev_skewnormal_refit_summary.json"

    if not emp_path.is_file() and args.ensure_empirical_json:
        subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts/build_empirical_admissible_draw_benchmark_report.py"),
                "--heldout-run",
                str(run_dir),
            ],
            check=True,
        )
    if not emp_path.is_file():
        raise FileNotFoundError(f"Missing {emp_path.name}")
    if not sk_path.is_file():
        raise FileNotFoundError(f"Missing {sk_path.name}; run build_ev_skewnormal_refit_evaluation.py --heldout-run {run_dir}")

    emp = json.loads(emp_path.read_text(encoding="utf-8"))
    sk = json.loads(sk_path.read_text(encoding="utf-8"))
    sks = sk["summary"]

    emp_ev = {
        "MAE": emp["point_accuracy_event_mean_prediction"]["MAE"]["EV"],
        "RMSE": emp["point_accuracy_event_mean_prediction"]["RMSE"]["EV"],
        "NLL": None,
        "COV50": emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]["EV"]["50"],
        "COV80": emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]["EV"]["80"],
        "COV90": emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]["EV"]["90"],
        "CRPS": emp["probabilistic_sample_based"]["mean_CRPS"]["EV"],
    }
    skew_ev = {
        "MAE": sks["MAE"],
        "RMSE": sks["RMSE"],
        "NLL": sks["NLL"],
        "COV50": sks["COV50"],
        "COV80": sks["COV80"],
        "COV90": sks["COV90"],
        "CRPS": sks["CRPS"],
    }

    ng_path = args.ngboost_scores.resolve()
    if not ng_path.is_file():
        raise FileNotFoundError(ng_path)
    ng_df = pd.read_csv(ng_path)
    held_ids = set()
    summ_pq = run_dir / "predictive_summary_by_event.parquet"
    summ_csv = run_dir / "predictive_summary_by_event.csv"
    if summ_pq.is_file():
        held_ids = set(pd.read_parquet(summ_pq, columns=["event_id"])["event_id"].astype(int))
    elif summ_csv.is_file():
        held_ids = set(pd.read_csv(summ_csv, usecols=["event_id"])["event_id"].astype(int))
    else:
        held_ids = set(ng_df["event_id"].astype(int))

    ng_df = ng_df.loc[ng_df["event_id"].isin(held_ids)].copy()
    if len(ng_df) != len(held_ids):
        raise ValueError(f"NGBoost rows {len(ng_df)} != held-out events {len(held_ids)}")
    ng_ev = agg_ngboost_ev(ng_df)

    keys = ["MAE", "RMSE", "NLL", "COV50", "COV80", "COV90", "CRPS"]

    def row(d: dict) -> dict:
        return {k: d[k] for k in keys}

    out = {
        "heldout_run": str(run_dir),
        "n_events_held_out": emp.get("n_events"),
        "admissible_draws_per_event_median": emp.get("draws_per_event_expected"),
        "ngboost_scores_file": str(ng_path),
        "n_events_ngboost_used": ng_ev["n_events"],
        "empirical_admissible_draws_EV": row(emp_ev),
        "skew_normal_refit_EV": row(skew_ev),
        "ngboost_gaussian_EV": {k: ng_ev[k] for k in keys},
        "notes": {
            "empirical": "Sample mean, sample quantiles, empirical CRPS; no density → NLL omitted.",
            "skew_normal": sk.get("method", "skewnorm.fit per event on admissible EV draws."),
            "ngboost": "Independent Normal per target from ngboost; EV marginals only here.",
        },
    }

    json_path = run_dir / "ev_diagnostics_empirical_skew_ngboost.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def fmt(x: float | None) -> str:
        if x is None:
            return "—"
        return f"{float(x):.6f}"

    lines = [
        "# EV diagnostics: empirical draws vs skew-normal refit vs NGBoost",
        "",
        f"- **Held-out run:** `{run_dir.name}`",
        f"- **Events:** {out['n_events_held_out']}",
        f"- **Admissible EV draws per event (median):** {out['admissible_draws_per_event_median']}",
        f"- **NGBoost file:** `{ng_path.relative_to(_ROOT)}`",
        "",
        "All values are **nan-mean over held-out events** (same `event_id` set for all three columns).",
        "",
        "| Metric | Empirical (admissible draws) | Skew-normal refit | NGBoost (EV) |",
        "|--------|-----------------------------:|------------------:|-------------:|",
    ]
    for k in keys:
        lines.append(
            f"| {k} | {fmt(emp_ev[k])} | {fmt(skew_ev[k])} | {fmt(ng_ev[k])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- **Empirical:** {out['notes']['empirical']}",
            f"- **Skew-normal:** {out['notes']['skew_normal']}",
            f"- **NGBoost:** {out['notes']['ngboost']}",
            "",
            "## JSON",
            "",
            f"- `{json_path.name}`",
            "",
        ]
    )
    md_path = run_dir / "ev_diagnostics_empirical_skew_ngboost.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True, "md": str(md_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
