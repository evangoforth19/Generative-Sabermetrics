#!/usr/bin/env python3
"""Validate processed SBI datasets: schema, splits, leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src import schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = ap.parse_args()
    proc = args.project_root / "data_processed"
    man = args.project_root / "manifests"
    errors: list[str] = []

    split_events = pd.read_parquet(proc / "event_split_table.parquet")
    ev_sets = {
        s: set(split_events.loc[split_events["split"].eq(s), "event_id"])
        for s in ("train", "calibration", "test")
    }
    inter = ev_sets["train"] & ev_sets["calibration"]
    if inter:
        errors.append(f"train/calibration event overlap: {len(inter)}")
    inter = ev_sets["train"] & ev_sets["test"]
    if inter:
        errors.append(f"train/test event overlap: {len(inter)}")
    inter = ev_sets["calibration"] & ev_sets["test"]
    if inter:
        errors.append(f"calibration/test event overlap: {len(inter)}")

    for name in ("u_train", "z_train", "joint_train"):
        df = pd.read_parquet(proc / f"{name}.parquet")
        if "split" in df.columns and not df["split"].eq("train").all():
            errors.append(f"{name}: contains non-train rows")

    for tbl, spl in [
        ("u_calibration", "calibration"),
        ("u_test", "test"),
        ("z_calibration", "calibration"),
        ("z_test", "test"),
    ]:
        df = pd.read_parquet(proc / f"{tbl}.parquet")
        if not df["split"].eq(spl).all():
            errors.append(f"{tbl}: split column inconsistent")

    stats_path = man / "standardization_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"standardization stats for {len(stats)} columns")
    else:
        errors.append("missing standardization_stats.json")

    u_tr = pd.read_parquet(proc / "u_train.parquet")
    z_tr = pd.read_parquet(proc / "z_train.parquet")
    z_cal = pd.read_parquet(proc / "z_calibration.parquet")
    z_te = pd.read_parquet(proc / "z_test.parquet")

    ev_u = set(u_tr["event_id"].unique())
    if not ev_u.issubset(ev_sets["train"]):
        errors.append("u_train has event_ids outside train split")
    ev_z_tr = set(z_tr["event_id"].unique())
    ev_z_cal = set(z_cal["event_id"].unique())
    ev_z_te = set(z_te["event_id"].unique())
    if not ev_z_tr.issubset(ev_sets["train"]):
        errors.append("z_train has event_ids outside train split")
    if not ev_z_cal.issubset(ev_sets["calibration"]):
        errors.append("z_calibration has event_ids outside calibration split")
    if not ev_z_te.issubset(ev_sets["test"]):
        errors.append("z_test has event_ids outside test split")

    for c in schema.U_COLUMNS:
        if c not in u_tr.columns:
            errors.append(f"u_train missing required u-column {c!r}")
    if "release_speed" not in u_tr.columns:
        errors.append("u_train missing release_speed")
    for trig in ("spin_axis_sin", "spin_axis_cos"):
        if trig not in u_tr.columns:
            errors.append(f"u_train missing {trig!r}")
    weight_candidates = [
        "uniform_draw_weight_within_event",
        "normalized_log_target_weight",
        "event_quality_weight",
        "combined_training_weight",
    ]
    if not any(w in u_tr.columns for w in weight_candidates):
        errors.append(f"u_train missing draw/event weight columns (expected one of {weight_candidates})")
    if "phi_star" in u_tr.columns:
        errors.append("u_train must not include phi_star (spray draw field); stage u uses d_tilde for spray direction")

    for c in schema.Z_TARGET_COLUMNS:
        if c not in z_tr.columns:
            errors.append(f"z_train missing required stage-z target column {c!r}")
    if "theta" in z_tr.columns:
        errors.append("z_train must not include ambiguous column `theta`")
    if "phi_star" in z_tr.columns:
        errors.append("z_train must not include phi_star (do not treat spray as launch angle)")

    g = [c for c in schema.G_COLUMNS if c in u_tr.columns]
    print("resolved g in u_train:", len(g), "/", len(schema.G_COLUMNS))
    miss_g = [c for c in schema.G_COLUMNS if c not in u_tr.columns]
    if miss_g:
        errors.append(f"u_train missing g columns {miss_g}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("OK: split integrity checks passed.")


if __name__ == "__main__":
    main()
