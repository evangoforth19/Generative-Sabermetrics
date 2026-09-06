#!/usr/bin/env python3
"""
Export **every balls-in-play (BIP) Statcast row** (full columns) for the twelve SBI production
hitters from the **pybaseball weekly shard download** (2020 through whatever shards exist).

This is **not** every pitch type (no balls/strikes/fouls)—only rows with ``type == 'X'`` (in play),
for batters whose MLBAM ids appear in ``selected_events.parquet``.

Reads all ``data/statcast_pybaseball/chunks/statcast_*.parquet`` shards (canonical full download).
Optionally ``--also-scan-combined`` merges in any BIP rows from ``statcast_all.parquet`` that are
not already present (dedupe on game_pk, at_bat_number, pitch_number if available).

Outputs:
  - Parquet with full Statcast schema + ``batter_name`` column
  - JSON manifest with row counts per hitter and shard coverage

Usage:
  python scripts/export_twelve_hitters_bip_full_statcast.py
  python scripts/export_twelve_hitters_bip_full_statcast.py --output data/statcast_pybaseball/twelve_hitters_bip_full.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHUNKS = ROOT / "data" / "statcast_pybaseball" / "chunks"
DEFAULT_COMBINED = ROOT / "data" / "statcast_pybaseball" / "statcast_all.parquet"
DEFAULT_SELECTED = (
    ROOT / "MCMC 2" / "Outputs" / "refactor_exact_root_rhh" / "production" / "selected_events.parquet"
)
DEFAULT_OUT = ROOT / "data" / "statcast_pybaseball" / "twelve_hitters_bip_full_statcast.parquet"
DEFAULT_MANIFEST = ROOT / "data" / "statcast_pybaseball" / "twelve_hitters_bip_full_statcast_manifest.json"


def _dedupe_keys(df: pd.DataFrame) -> pd.Series:
    if all(c in df.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
        return (
            df["game_pk"].astype(str)
            + "_"
            + df["at_bat_number"].astype(str)
            + "_"
            + df["pitch_number"].astype(str)
        )
    return pd.RangeIndex(len(df)).astype(str)


def _filter_bip_twelve(df: pd.DataFrame, id_set: set[int], name_by_id: dict[int, str]) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["type"].astype(str).str.upper()
    bid = pd.to_numeric(df["batter"], errors="coerce")
    m = t.eq("X") & bid.notna() & bid.astype(int).isin(id_set)
    out = df.loc[m].copy()
    if out.empty:
        return out
    out["batter_name"] = pd.to_numeric(out["batter"], errors="coerce").astype(int).map(name_by_id)
    out = out.loc[out["batter_name"].notna()].copy()
    if "game_date" in out.columns:
        gd = pd.to_datetime(out["game_date"], errors="coerce")
        out = out.loc[gd >= pd.Timestamp("2020-01-01")].copy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--selected-events", type=Path, default=DEFAULT_SELECTED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--also-scan-combined",
        action="store_true",
        help="Merge BIP rows from statcast_all.parquet not duplicated in shard scan.",
    )
    p.add_argument("--combined-path", type=Path, default=DEFAULT_COMBINED)
    args = p.parse_args()

    sel = pd.read_parquet(args.selected_events.resolve(), columns=["batter", "batter_name"])
    sel["batter_name"] = sel["batter_name"].astype(str).str.strip().str.lower()
    id_set = set(pd.to_numeric(sel["batter"], errors="coerce").dropna().astype(int).tolist())
    name_by_id = (
        sel.drop_duplicates("batter").set_index("batter")["batter_name"].astype(str).str.strip().str.lower().to_dict()
    )

    chunks_dir = args.chunks_dir.resolve()
    if not chunks_dir.is_dir():
        raise SystemExit(f"Missing chunks dir: {chunks_dir}")

    paths = sorted(chunks_dir.glob("statcast_*.parquet"))
    if not paths:
        raise SystemExit(f"No statcast_*.parquet under {chunks_dir}")

    pieces: list[pd.DataFrame] = []
    shard_stats: list[dict] = []
    seen_keys: set[str] | None = None
    if args.also_scan_combined:
        seen_keys = set()

    for path in paths:
        df = pd.read_parquet(path)
        n_in = len(df)
        sub = _filter_bip_twelve(df, id_set, name_by_id)
        shard_stats.append({"shard": path.name, "rows_in": n_in, "bip_twelve_rows": len(sub)})
        if not sub.empty:
            if seen_keys is not None:
                k = _dedupe_keys(sub)
                mask = ~k.astype(str).isin(seen_keys)
                sub = sub.loc[mask].copy()
                for kk in k[mask].astype(str):
                    seen_keys.add(str(kk))
            if not sub.empty:
                pieces.append(sub)

    if args.also_scan_combined and args.combined_path.is_file():
        cdf = pd.read_parquet(args.combined_path.resolve())
        extra = _filter_bip_twelve(cdf, id_set, name_by_id)
        if not extra.empty and seen_keys is not None:
            k = _dedupe_keys(extra)
            mask = ~k.astype(str).isin(seen_keys)
            extra = extra.loc[mask].copy()
        if not extra.empty:
            pieces.append(extra)
            shard_stats.append({"shard": str(args.combined_path.name), "rows_in": len(cdf), "bip_twelve_rows": len(extra)})

    if not pieces:
        raise SystemExit("No BIP rows found for the twelve hitters; check chunks and selected_events.")

    out = pd.concat(pieces, ignore_index=True)
    # Final dedupe on keys if columns exist
    if all(c in out.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
        out = out.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="first")

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    counts = (
        out.groupby("batter_name", dropna=False)
        .size()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "chunks_dir": str(chunks_dir),
        "n_shards_scanned": len(paths),
        "also_scan_combined": bool(args.also_scan_combined),
        "selected_events": str(args.selected_events.resolve()),
        "output_parquet": str(out_path),
        "n_rows": int(len(out)),
        "n_columns": int(out.shape[1]),
        "rows_by_batter_name": counts,
        "shard_stats": shard_stats,
    }

    args.manifest.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} rows={len(out):,} cols={out.shape[1]}")
    print(json.dumps(counts, indent=2))
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
