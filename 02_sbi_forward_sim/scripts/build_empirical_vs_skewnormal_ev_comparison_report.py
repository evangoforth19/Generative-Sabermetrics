#!/usr/bin/env python3
"""
Side-by-side EV predictive diagnostics for a held-out run:
  - **Empirical:** admissible draws only (same definitions as run_heldout_pipeline_test /
    empirical_admissible_draw_benchmark.json).
  - **Skew-normal:** per-event scipy.stats.skewnorm.fit on admissible EV draws
    (ev_skewnormal_refit_summary.json).

Writes ``predictive_diagnostics_empirical_vs_skewnormal_EV.{md,json}`` under --heldout-run.

Example:
  python scripts/build_empirical_vs_skewnormal_ev_comparison_report.py \\
    --heldout-run outputs/heldout_pipeline_test/20260409_175049Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-run", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.heldout_run.resolve()
    emp_p = run_dir / "empirical_admissible_draw_benchmark.json"
    sk_p = run_dir / "ev_skewnormal_refit_summary.json"
    if not emp_p.is_file():
        raise FileNotFoundError(
            f"Missing {emp_p.name}; run scripts/build_empirical_admissible_draw_benchmark_report.py first."
        )
    if not sk_p.is_file():
        raise FileNotFoundError(
            f"Missing {sk_p.name}; run reports/.../build_ev_skewnormal_refit_evaluation.py --heldout-run ... first."
        )

    emp = json.loads(emp_p.read_text(encoding="utf-8"))
    sk = json.loads(sk_p.read_text(encoding="utf-8"))
    e_ev = emp["point_accuracy_event_mean_prediction"]
    p_ev = emp["probabilistic_sample_based"]["mean_CRPS"]["EV"]
    c_ev = emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]["EV"]

    emp_row = {
        "MAE": e_ev["MAE"]["EV"],
        "RMSE": e_ev["RMSE"]["EV"],
        "NLL": None,
        "COV50": c_ev["50"],
        "COV80": c_ev["80"],
        "COV90": c_ev["90"],
        "CRPS": p_ev,
    }
    sk_sum = sk["summary"]
    sk_row = {
        "MAE": sk_sum["MAE"],
        "RMSE": sk_sum["RMSE"],
        "NLL": sk_sum["NLL"],
        "COV50": sk_sum["COV50"],
        "COV80": sk_sum["COV80"],
        "COV90": sk_sum["COV90"],
        "CRPS": sk_sum["CRPS"],
    }

    delta = {}
    for k in ("MAE", "RMSE", "COV50", "COV80", "COV90", "CRPS"):
        delta[k] = float(sk_row[k]) - float(emp_row[k])
    delta["NLL"] = None

    out = {
        "heldout_run": str(run_dir),
        "n_events": emp.get("n_events"),
        "admissible_draws_per_event_median": emp.get("draws_per_event_expected"),
        "empirical_admissible_draws": {
            "description": "Point: error of sample mean vs observed. Probabilistic: sample quantiles, empirical CRPS. No density → no NLL.",
            "metrics_EV": emp_row,
        },
        "skew_normal_refit_on_admissible_EV": {
            "description": sk.get("method", "skewnorm.fit per event"),
            "crps_method": sk_sum.get("crps_method"),
            "n_events_skewnorm": sk_sum.get("n_events_skewnorm"),
            "n_events_gaussian_fallback": sk_sum.get("n_events_gaussian_fallback"),
            "metrics_EV": sk_row,
        },
        "delta_skew_minus_empirical_EV": delta,
        "LA_SA_empirical_only": {
            "MAE": {t: emp["point_accuracy_event_mean_prediction"]["MAE"][t] for t in ("LA", "SA")},
            "RMSE": {t: emp["point_accuracy_event_mean_prediction"]["RMSE"][t] for t in ("LA", "SA")},
            "CRPS": {t: emp["probabilistic_sample_based"]["mean_CRPS"][t] for t in ("LA", "SA")},
            "COV50": {
                t: emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"][t]["50"]
                for t in ("LA", "SA")
            },
            "COV80": {
                t: emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"][t]["80"]
                for t in ("LA", "SA")
            },
            "COV90": {
                t: emp["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"][t]["90"]
                for t in ("LA", "SA")
            },
            "note": "No skew-normal refit in repo for LA/SA; only empirical admissible draws.",
        },
    }

    json_path = run_dir / "predictive_diagnostics_empirical_vs_skewnormal_EV.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def fmt(x: float | None) -> str:
        if x is None:
            return "—"
        return f"{x:.6f}"

    def fmt_d(x: float | None) -> str:
        if x is None:
            return "—"
        return f"{x:+.6f}"

    lines = [
        "# Predictive diagnostics: empirical admissible draws vs skew-normal (EV only)",
        "",
        f"- **Run:** `{run_dir.name}`",
        f"- **Events:** {out['n_events']}",
        f"- **Admissible EV draws per event (median):** {out['admissible_draws_per_event_median']}",
        "",
        "## Methods",
        "",
        "| Column | Definition |",
        "|--------|------------|",
        "| **Empirical** | Predictive mean = mean of 250 admissible EV draws; intervals from **sample** quantiles; CRPS = **empirical** CRPS on those draws. |",
        "| **Skew-normal** | Per-event `scipy.stats.skewnorm.fit` on the same 250 draws; mean / quantiles / CRPS / NLL from the **fitted** law. |",
        "",
        "**NLL:** only defined for the skew-normal (or another density); omitted for the empirical column.",
        "",
        "## EV comparison (nan-mean over 1010 events)",
        "",
        "| Metric | Empirical (no fit) | Skew-normal fit | Δ (skew − empirical) |",
        "|--------|-------------------:|----------------:|---------------------:|",
        f"| MAE | {fmt(emp_row['MAE'])} | {fmt(sk_row['MAE'])} | {fmt_d(delta['MAE'])} |",
        f"| RMSE | {fmt(emp_row['RMSE'])} | {fmt(sk_row['RMSE'])} | {fmt_d(delta['RMSE'])} |",
        f"| NLL | {fmt(emp_row['NLL'])} | {fmt(sk_row['NLL'])} | {fmt_d(delta['NLL'])} |",
        f"| COV50 | {fmt(emp_row['COV50'])} | {fmt(sk_row['COV50'])} | {fmt_d(delta['COV50'])} |",
        f"| COV80 | {fmt(emp_row['COV80'])} | {fmt(sk_row['COV80'])} | {fmt_d(delta['COV80'])} |",
        f"| COV90 | {fmt(emp_row['COV90'])} | {fmt(sk_row['COV90'])} | {fmt_d(delta['COV90'])} |",
        f"| CRPS | {fmt(emp_row['CRPS'])} | {fmt(sk_row['CRPS'])} | {fmt_d(delta['CRPS'])} |",
        "",
        "### Interpretation (short)",
        "",
        "- **MAE/RMSE** use different predictive means (sample mean vs fitted skew-normal mean); deltas are small here.",
        "- **Coverage** moves toward nominal 0.5 / 0.8 / 0.9 when the parametric fit smooths tail behavior vs raw 250-point empirical intervals.",
        "- **CRPS** can improve under the fit if the fitted law is a better scoring rule for the observations than the discrete empirical measure.",
        "",
        "## LA / SA (empirical admissible draws only)",
        "",
        "Skew-normal refit is implemented for **EV** only. LA and SA below are the same **empirical** construction as the forward pipeline.",
        "",
        "| Target | MAE | RMSE | CRPS | COV50 | COV80 | COV90 |",
        "|--------|----:|-----:|-----:|------:|------:|------:|",
    ]
    lo = out["LA_SA_empirical_only"]
    for tgt in ("LA", "SA"):
        lines.append(
            f"| {tgt} | {fmt(lo['MAE'][tgt])} | {fmt(lo['RMSE'][tgt])} | {fmt(lo['CRPS'][tgt])} | "
            f"{fmt(lo['COV50'][tgt])} | {fmt(lo['COV80'][tgt])} | {fmt(lo['COV90'][tgt])} |"
        )
    lines.extend(
        [
            "",
            "## Source files",
            "",
            f"- Empirical: `{emp_p.name}`",
            f"- Skew-normal: `{sk_p.name}`",
            f"- This report JSON: `{json_path.name}`",
            "",
        ]
    )

    md_path = run_dir / "predictive_diagnostics_empirical_vs_skewnormal_EV.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True, "md": str(md_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
