#!/usr/bin/env python3
"""Sweep downstream support/temperature policies for a stage-u checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMC2_ROOT = PROJECT_ROOT.parent
REPO_ROOT = MMC2_ROOT.parent


@dataclass
class Candidate:
    candidate_id: str
    policy: str
    t_ang_multiplier: float


def _pit_uniformity(counts: list[int]) -> dict[str, float]:
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0 or c.sum() <= 0:
        return {"chi_per_bin": float("nan"), "mean_abs_bin_prob_dev": float("nan")}
    exp = c.sum() / c.size
    return {
        "chi_per_bin": float(np.sum((c - exp) ** 2 / max(exp, 1e-12)) / c.size),
        "mean_abs_bin_prob_dev": float(np.mean(np.abs(c / c.sum() - 1.0 / c.size))),
    }


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    crps = m["probabilistic_sample_based"]["mean_CRPS"]
    cov = m["probabilistic_sample_based"]["coverage_rate_marginal_sample_central"]
    point = m["point_accuracy_event_mean_prediction"]
    tail = m.get("tail_diagnostics", {}).get("global", {})
    pit = m.get("pit_histogram_counts", {})
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_events": m.get("n_events"),
        "admissible_fraction": m.get("mean_admissible_fraction"),
    }
    for tgt in ("EV", "LA", "SA"):
        row[f"CRPS_{tgt}"] = crps.get(tgt)
        row[f"MAE_{tgt}"] = point["MAE"].get(tgt)
        row[f"RMSE_{tgt}"] = point["RMSE"].get(tgt)
        row[f"bias_{tgt}"] = point["bias"].get(tgt)
        for level in ("50", "80", "90"):
            row[f"coverage_{tgt}_{level}"] = cov[tgt].get(level)
        pu = _pit_uniformity(pit.get(tgt, []))
        row[f"PIT_chi_per_bin_{tgt}"] = pu["chi_per_bin"]
        row[f"PIT_mad_{tgt}"] = pu["mean_abs_bin_prob_dev"]
    for block_name in ("EV", "u_v_ss_tilde", "SA"):
        block = tail.get(block_name, {})
        for key in ("p99", "p99_9", "max"):
            row[f"tail_{block_name}_{key}"] = block.get(key)
        for thr, val in block.get("threshold_rates", {}).items():
            row[f"tail_{block_name}_rate_gt_{thr}"] = val
    return row


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Small markdown table helper; avoids pandas' optional tabulate dependency."""
    sub = df.loc[:, cols].copy()
    headers = list(sub.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in sub.iterrows():
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _extract_output_dir(log_path: Path) -> Path:
    out_dir: Path | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Done. "):
            raw = line.replace("Done.", "", 1).strip()
            out_dir = Path(raw)
    if out_dir is None:
        raise RuntimeError(f"Could not find output directory in {log_path}")
    return out_dir


def _candidate_command(
    python_exe: Path,
    candidate: Candidate,
    *,
    u_run_dir: Path,
    z_run_dir: Path,
    n_samples: int,
    seed: int,
    max_events: int | None,
) -> list[str]:
    cmd = [
        str(python_exe),
        str(PROJECT_ROOT / "scripts" / "run_heldout_pipeline_test.py"),
        "--u-run-dir",
        str(u_run_dir),
        "--z-run-dir",
        str(z_run_dir),
        "--n-samples",
        str(n_samples),
        "--seed",
        str(seed),
        "--no-event-progress",
        "--u-t-ang-multiplier",
        str(candidate.t_ang_multiplier),
        "--report-filename",
        f"downstream_support_{candidate.candidate_id}.md",
    ]
    if candidate.policy == "u_d_reject":
        cmd += ["--u-d-tilde-limits", "-45", "45"]
    elif candidate.policy == "decoded_sa_filter":
        cmd += ["--decoded-sa-limits", "-45", "45"]
    else:
        raise ValueError(f"Unknown policy {candidate.policy!r}")
    if max_events is not None:
        cmd += ["--max-events", str(max_events)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "heldout_pipeline_test" / "20260526_171808Z",
    )
    ap.add_argument(
        "--u-run-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "p_u_given_g" / "20260526_165540Z",
    )
    ap.add_argument(
        "--z-run-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "p_z_given_u_g" / "20260501_055549Z",
    )
    ap.add_argument(
        "--python",
        type=Path,
        default=MMC2_ROOT / ".venv_prod" / "bin" / "python",
    )
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sweep_root = PROJECT_ROOT / "outputs" / "downstream_support_tuning" / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(f"{policy}_Tang{str(mult).replace('.', 'p')}", policy, mult)
        for policy in ("u_d_reject", "decoded_sa_filter")
        for mult in (0.75, 1.0, 1.25, 1.5)
    ]

    active: list[tuple[Candidate, subprocess.Popen, Path]] = []
    completed: list[dict[str, Any]] = []

    def launch(c: Candidate) -> None:
        log_path = sweep_root / f"{c.candidate_id}.log"
        cmd = _candidate_command(
            args.python,
            c,
            u_run_dir=args.u_run_dir,
            z_run_dir=args.z_run_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            max_events=args.max_events,
        )
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
        fh = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        active.append((c, proc, log_path))
        time.sleep(1.1)

    pending = list(candidates)
    while pending or active:
        while pending and len(active) < max(1, int(args.max_parallel)):
            launch(pending.pop(0))
        time.sleep(5)
        still: list[tuple[Candidate, subprocess.Popen, Path]] = []
        for c, proc, log_path in active:
            rc = proc.poll()
            if rc is None:
                still.append((c, proc, log_path))
                continue
            if rc != 0:
                raise RuntimeError(f"Candidate {c.candidate_id} failed with exit code {rc}; see {log_path}")
            run_dir = _extract_output_dir(log_path)
            row = {
                "candidate_id": c.candidate_id,
                "policy": c.policy,
                "T_ang_multiplier": c.t_ang_multiplier,
                "log_path": str(log_path),
                **_read_metrics(run_dir),
            }
            completed.append(row)
        active = still

    baseline = {"candidate_id": "baseline_old_frozen", "policy": "baseline", "T_ang_multiplier": 1.0}
    baseline.update(_read_metrics(args.baseline_run_dir.resolve()))
    rows = [baseline] + completed
    df = pd.DataFrame(rows)
    df.to_csv(sweep_root / "downstream_support_sweep_summary.csv", index=False)

    base = baseline
    for tgt in ("EV", "LA", "SA"):
        df[f"delta_CRPS_{tgt}_vs_baseline"] = df[f"CRPS_{tgt}"] - base[f"CRPS_{tgt}"]
        for level in ("50", "80", "90"):
            df[f"abs_cov_error_{tgt}_{level}"] = (df[f"coverage_{tgt}_{level}"] - float(level) / 100.0).abs()
    df["rank_score"] = (
        df["delta_CRPS_EV_vs_baseline"].clip(lower=0)
        + df["delta_CRPS_LA_vs_baseline"].clip(lower=0)
        + df["delta_CRPS_SA_vs_baseline"].clip(lower=0)
        + df["abs_cov_error_SA_80"]
        + df["abs_cov_error_SA_90"]
    )
    ranked = df.sort_values("rank_score")
    ranked.to_csv(sweep_root / "downstream_support_sweep_ranked.csv", index=False)

    report_lines = [
        "# Downstream Support Tuning Sweep",
        "",
        f"Baseline: `{args.baseline_run_dir.resolve()}`",
        f"Stage-u candidate checkpoint: `{args.u_run_dir.resolve()}`",
        f"Stage-z checkpoint: `{args.z_run_dir.resolve()}`",
        f"Samples/event: `{args.n_samples}`",
        "",
        "## Top Candidates",
        "",
    ]
    show_cols = [
        "candidate_id",
        "policy",
        "T_ang_multiplier",
        "CRPS_EV",
        "CRPS_LA",
        "CRPS_SA",
        "coverage_SA_50",
        "coverage_SA_80",
        "coverage_SA_90",
        "tail_EV_p99_9",
        "tail_EV_max",
        "tail_u_v_ss_tilde_p99_9",
        "tail_u_v_ss_tilde_max",
        "rank_score",
        "run_dir",
    ]
    report_lines.append(_markdown_table(ranked.head(12), show_cols))
    report_lines.append("")
    report_lines.append("Artifacts:")
    report_lines.append("- `downstream_support_sweep_summary.csv`")
    report_lines.append("- `downstream_support_sweep_ranked.csv`")
    (sweep_root / "downstream_support_sweep_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("Downstream support sweep complete:", sweep_root)


if __name__ == "__main__":
    main()
