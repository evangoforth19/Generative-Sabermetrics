#!/usr/bin/env python3
"""
Build empirical BIP pitch-context pools for SBI ``g`` (and decoder extras) from the
**combined pybaseball Statcast download** (typically 2020–present / through download ``--end``).

Canonical combined file (after ``python scripts/download_statcast_pybaseball.py --combine-only``):

  ``<repo>/data/statcast_pybaseball/statcast_all.parquet``

This script:
  1. Reads ``statcast_all.parquet`` (or ``--statcast-path``).
  2. Restricts to the **twelve** ``batter`` MLBAM ids appearing in production ``selected_events``
     (same RHH SBI cohort as ``refactor_exact_root_rhh`` by default).
  3. Keeps **balls in play** rows: ``type == 'X'``, ``launch_speed >= 40`` (mph).
  4. Engineers ``z_count``, ``spin_axis_sin``, ``spin_axis_cos`` like ``sbi_forward_sim`` / dataset
     builders (``z_count_from_balls_strikes``, ``add_spin_axis_trig``).
  5. Ensures ``release_speed`` exists (uses Statcast column; if missing, derives from ``vx0,vy0,vz0``).
  6. Writes a single Parquet pool + JSON manifest (row counts per ``batter_name``) for downstream
     **with-replacement bootstrap** of ``g`` rows **per hitter**.

Downstream simulators should sample **within** ``batter_name == <player>`` so resampled pitches
match that hitter’s realized BIP pitch mix.

Usage:
  python scripts/build_sbi_twelve_bip_g_bootstrap_pool.py
  python scripts/build_sbi_twelve_bip_g_bootstrap_pool.py \\
      --statcast-path data/statcast_pybaseball/statcast_all.parquet \\
      --selected-events \"MCMC 2/Outputs/refactor_exact_root_rhh/production/selected_events.parquet\" \\
      --output-dir outputs/sbi_bootstrap_g_pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SBI_ROOT = ROOT / "MCMC 2" / "sbi_forward_sim"
sys.path.insert(0, str(SBI_ROOT))

from src.feature_engineering import (  # noqa: E402
    add_spin_axis_trig,
    attach_release_speed_from_kinematics,
    z_count_from_balls_strikes,
)
from src.schema import G_COLUMNS  # noqa: E402

DEFAULT_STATCAST = ROOT / "data" / "statcast_pybaseball" / "statcast_all.parquet"
DEFAULT_SELECTED = (
    ROOT / "MCMC 2" / "Outputs" / "refactor_exact_root_rhh" / "production" / "selected_events.parquet"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "sbi_bootstrap_g_pool"

# Pool columns: neural g + lineage / bootstrap keys + decoder-required extra not in G_COLUMNS.
POOL_EXTRA = ["batter", "batter_name", "spin_axis", "balls", "strikes", "release_pos_y", "game_date"]


def _ensure_release_speed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "release_speed" in out.columns and out["release_speed"].notna().any():
        return out
    return attach_release_speed_from_kinematics(out, table_name="statcast_bip_pool")


def main() -> None:
    p = argparse.ArgumentParser(description="Build BIP g bootstrap pools from statcast_all.parquet.")
    p.add_argument("--statcast-path", type=Path, default=DEFAULT_STATCAST)
    p.add_argument("--selected-events", type=Path, default=DEFAULT_SELECTED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min-launch-speed", type=float, default=40.0)
    args = p.parse_args()

    sc_path = args.statcast_path.resolve()
    if not sc_path.is_file():
        raise SystemExit(
            f"Missing {sc_path}. Build it with:\n"
            "  python scripts/download_statcast_pybaseball.py --combine-only "
            f"--output-dir {sc_path.parent}"
        )

    sel_path = args.selected_events.resolve()
    if not sel_path.is_file():
        raise SystemExit(f"Missing selected_events: {sel_path}")

    selected = pd.read_parquet(sel_path, columns=["batter", "batter_name"])
    selected["batter_name"] = selected["batter_name"].astype(str).str.strip().str.lower()
    id_set = set(pd.to_numeric(selected["batter"], errors="coerce").dropna().astype(int).tolist())
    name_by_id = (
        selected.drop_duplicates("batter")
        .set_index("batter")["batter_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_dict()
    )

    use_cols = None  # full table; ~280k rows is acceptable
    df = pd.read_parquet(sc_path, columns=use_cols)
    df = df.loc[pd.to_numeric(df["batter"], errors="coerce").astype("Int64").isin(list(id_set))].copy()
    df = df.loc[df["type"].astype(str).str.upper().eq("X")].copy()
    ls = pd.to_numeric(df["launch_speed"], errors="coerce")
    df = df.loc[ls.notna() & (ls >= float(args.min_launch_speed))].copy()

    df["batter_name"] = pd.to_numeric(df["batter"], errors="coerce").astype(int).map(name_by_id)
    bad = df["batter_name"].isna()
    if bad.any():
        df = df.loc[~bad].copy()

    df = _ensure_release_speed(df)
    df = add_spin_axis_trig(df, axis_col="spin_axis")
    df = z_count_from_balls_strikes(df, table_name="statcast_bip_pool")

    missing = [c for c in G_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"After transforms, still missing G_COLUMNS: {missing}")

    miss_dec = [c for c in ("release_pos_y",) if c not in df.columns]
    if miss_dec:
        raise SystemExit(f"Missing decoder context columns: {miss_dec}")

    extra = [c for c in POOL_EXTRA if c in df.columns]
    out_cols = list(G_COLUMNS) + [c for c in extra if c not in G_COLUMNS]
    out = df[out_cols].copy()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pq = out_dir / "bip_g_pool_twelve_hitters.parquet"
    out.to_parquet(out_pq, index=False)

    counts = (
        out.groupby("batter_name", dropna=False)
        .size()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "statcast_path": str(sc_path),
        "selected_events_path": str(sel_path),
        "n_unique_batter_ids": len(id_set),
        "min_launch_speed_mph": float(args.min_launch_speed),
        "filter_type": "X",
        "g_columns": list(G_COLUMNS),
        "pool_columns": out_cols,
        "output_parquet": str(out_pq),
        "n_rows_total": int(len(out)),
        "rows_by_batter_name": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_pq} rows={len(out):,} cols={len(out_cols)}")
    print("Rows by batter_name:")
    for k in sorted(counts, key=lambda x: str(x)):
        print(f"  {k}: {counts[k]:,}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
