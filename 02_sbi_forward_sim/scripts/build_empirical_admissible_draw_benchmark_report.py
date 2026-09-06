#!/usr/bin/env python3
"""
Recompute held-out benchmarks from admissible predictive draws only (no parametric fit).

Uses the same definitions as ``run_heldout_pipeline_test.py``:
  - MAE / RMSE / bias: error of **sample mean** vs observed y per event, then averaged.
  - CRPS: empirical CRPS from admissible samples.
  - Coverage: central intervals from **sample** quantiles.

NLL is **not** defined for a raw empirical measure without a density model; omitted here
(parametric NLL appears in skew-normal / NGBoost-style reports).

Example:
  python scripts/build_empirical_admissible_draw_benchmark_report.py \\
    --heldout-run outputs/heldout_pipeline_test/20260409_175049Z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_MMC2_ROOT = _HERE.parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.target_transform_z import wrap_deg  # noqa: E402

TARGETS = ("EV", "LA", "SA")


def _empirical_crps(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n < 2:
        return float("nan")
    e1 = np.mean(np.abs(s - y))
    e2 = np.mean(np.abs(s.reshape(-1, 1) - s.reshape(1, -1)))
    return float(e1 - 0.5 * e2)


def _coverage_width(samples: np.ndarray, y: float, qlo: float, qhi: float) -> tuple[bool, float]:
    s = np.asarray(samples, dtype=np.float64).ravel()
    if len(s) < 2:
        return False, float("nan")
    lo, hi = np.quantile(s, [qlo, qhi])
    inside = bool(lo <= y <= hi)
    return inside, float(hi - lo)


def _load_summary(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "predictive_summary_by_event.parquet"
    csv = run_dir / "predictive_summary_by_event.csv"
    if pq.is_file():
        return pd.read_parquet(pq)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No predictive_summary in {run_dir}")


def _load_draws(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "predictive_draws_admissible.parquet"
    csv = run_dir / "predictive_draws_admissible.csv"
    if pq.is_file():
        return pd.read_parquet(pq, columns=["event_id", "EV", "LA", "SA"])
    if csv.is_file():
        return pd.read_csv(csv, usecols=["event_id", "EV", "LA", "SA"])
    raise FileNotFoundError(f"No predictive_draws_admissible in {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-run", type=Path, required=True, help="Pipeline output directory")
    args = ap.parse_args()
    run_dir = args.heldout_run.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    df_ev = _load_summary(run_dir)
    df_draw = _load_draws(run_dir)

    mae_mean = {t: [] for t in TARGETS}
    rmse_mean = {t: [] for t in TARGETS}
    bias_mean = {t: [] for t in TARGETS}
    crps = {t: [] for t in TARGETS}
    cov50 = {t: [] for t in TARGETS}
    cov80 = {t: [] for t in TARGETS}
    cov90 = {t: [] for t in TARGETS}
    width50 = {t: [] for t in TARGETS}
    width80 = {t: [] for t in TARGETS}
    width90 = {t: [] for t in TARGETS}

    obs_cols = {"EV": "observed_EV", "LA": "observed_LA", "SA": "observed_SA"}

    for _, er in df_ev.iterrows():
        eid = int(er["event_id"])
        sub = df_draw.loc[df_draw["event_id"] == eid]
        for tgt in TARGETS:
            obs = float(er[obs_cols[tgt]])
            use_wrap = tgt == "SA"
            pred_mean = float(er[f"pred_{tgt}_mean"])
            if not np.isfinite(pred_mean):
                pred_mean = float("nan")
            err_mean = pred_mean - obs
            if use_wrap:
                err_mean = float(wrap_deg(err_mean))
            mae_mean[tgt].append(abs(err_mean))
            rmse_mean[tgt].append(err_mean**2)
            bias_mean[tgt].append(err_mean)

            samps = sub[tgt].to_numpy(dtype=np.float64)
            if len(samps) > 0:
                crps[tgt].append(_empirical_crps(samps, obs))
                i50, w50 = _coverage_width(samps, obs, 0.25, 0.75)
                cov50[tgt].append(float(i50))
                width50[tgt].append(w50)
                i80, w80 = _coverage_width(samps, obs, 0.1, 0.9)
                cov80[tgt].append(float(i80))
                width80[tgt].append(w80)
                i90, w90 = _coverage_width(samps, obs, 0.05, 0.95)
                cov90[tgt].append(float(i90))
                width90[tgt].append(w90)
            else:
                crps[tgt].append(np.nan)
                cov50[tgt].append(np.nan)
                cov80[tgt].append(np.nan)
                cov90[tgt].append(np.nan)
                width50[tgt].append(np.nan)
                width80[tgt].append(np.nan)
                width90[tgt].append(np.nan)

    n_events = int(len(df_ev))
    out = {
        "heldout_run": str(run_dir),
        "n_events": n_events,
        "method": "empirical_admissible_draws_only_no_parametric_fit",
        "nll_note": "NLL not defined for raw sample measure; use parametric refit if needed.",
        "draws_per_event_expected": int(df_draw.groupby("event_id").size().median()),
        "point_accuracy_event_mean_prediction": {
            "MAE": {t: float(np.nanmean(mae_mean[t])) for t in TARGETS},
            "RMSE": {t: float(np.sqrt(np.nanmean(rmse_mean[t]))) for t in TARGETS},
            "bias": {t: float(np.nanmean(bias_mean[t])) for t in TARGETS},
        },
        "probabilistic_sample_based": {
            "mean_CRPS": {t: float(np.nanmean(crps[t])) for t in TARGETS},
            "coverage_rate_marginal_sample_central": {
                t: {
                    "50": float(np.nanmean(cov50[t])),
                    "80": float(np.nanmean(cov80[t])),
                    "90": float(np.nanmean(cov90[t])),
                }
                for t in TARGETS
            },
            "mean_interval_width_50": {t: float(np.nanmean(width50[t])) for t in TARGETS},
            "mean_interval_width_80": {t: float(np.nanmean(width80[t])) for t in TARGETS},
            "mean_interval_width_90": {t: float(np.nanmean(width90[t])) for t in TARGETS},
        },
    }

    metrics_path = run_dir / "metrics.json"
    verify: dict[str, float] = {}
    if metrics_path.is_file():
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        for tgt in TARGETS:
            verify[f"diff_MAE_{tgt}"] = abs(
                out["point_accuracy_event_mean_prediction"]["MAE"][tgt]
                - prev["point_accuracy_event_mean_prediction"]["MAE"][tgt]
            )
            verify[f"diff_CRPS_{tgt}"] = abs(
                out["probabilistic_sample_based"]["mean_CRPS"][tgt]
                - prev["probabilistic_sample_based"]["mean_CRPS"][tgt]
            )
        out["verification_vs_metrics_json_max_abs_diff"] = verify

    json_path = run_dir / "empirical_admissible_draw_benchmark.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    p = out["point_accuracy_event_mean_prediction"]
    pr = out["probabilistic_sample_based"]
    cov = pr["coverage_rate_marginal_sample_central"]

    lines = [
        "# Empirical admissible-draw benchmark (no parametric fit)",
        "",
        f"- **Run directory:** `{run_dir}`",
        f"- **Events:** {n_events}",
        "- **Predictive law:** empirical distribution of **admissible** decoder draws per event (no skew-normal or other fit).",
        f"- **{out['nll_note']}**",
        "",
        "## EV (exit velocity)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| MAE (mean sample vs obs) | {p['MAE']['EV']:.6f} |",
        f"| RMSE | {p['RMSE']['EV']:.6f} |",
        f"| Mean CRPS (empirical) | {pr['mean_CRPS']['EV']:.6f} |",
        f"| COV50 | {cov['EV']['50']:.6f} |",
        f"| COV80 | {cov['EV']['80']:.6f} |",
        f"| COV90 | {cov['EV']['90']:.6f} |",
        "",
        "## LA / SA (same construction)",
        "",
        "| Target | MAE | RMSE | CRPS | COV50 | COV80 | COV90 |",
        "|--------|----:|-----:|-----:|------:|------:|------:|",
    ]
    for tgt in TARGETS:
        lines.append(
            f"| {tgt} | {p['MAE'][tgt]:.6f} | {p['RMSE'][tgt]:.6f} | {pr['mean_CRPS'][tgt]:.6f} | "
            f"{cov[tgt]['50']:.6f} | {cov[tgt]['80']:.6f} | {cov[tgt]['90']:.6f} |"
        )
    if verify:
        lines.extend(
            [
                "",
                "## Check vs `metrics.json`",
                "",
                "Max absolute differences (should be ~0):",
                "",
                "```json",
                json.dumps(verify, indent=2),
                "```",
            ]
        )
    lines.extend(["", "## JSON", "", f"- `{json_path.name}`", ""])

    md_path = run_dir / "empirical_admissible_draw_benchmark_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"ok": True, "json": str(json_path), "md": str(md_path), "n_events": n_events}, indent=2))


if __name__ == "__main__":
    main()
