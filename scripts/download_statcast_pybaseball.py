#!/usr/bin/env python3
"""
Download Statcast pitch-level data via pybaseball.statcast, from a start date through today.

Uses weekly (configurable) date windows, resumable Parquet shards, and pybaseball disk caching
so interrupted runs can continue without re-hitting Savant for completed days.

Usage:
  pip install pybaseball pandas pyarrow
  python scripts/download_statcast_pybaseball.py --output-dir data/statcast_pybaseball

  # Resume-friendly: skips shards that already exist unless --overwrite
  python scripts/download_statcast_pybaseball.py --start 2020-01-01 --end 2026-04-30

  # Merge shards into one Parquet (needs enough RAM / disk; uses pyarrow)
  python scripts/download_statcast_pybaseball.py --combine-only --output-dir data/statcast_pybaseball

  # After combine: build hitter clustering features (writes statcast_hitter_features_2020_2026/)
  python scripts/download_statcast_pybaseball.py --combine-only --post-hitter-features --output-dir data/statcast_pybaseball

  # After shards finish: rebuild Metric Folder / Full Data Set.parquet (BIP rows, G features)
  python scripts/download_statcast_pybaseball.py --start 2020-01-01 --output-dir data/statcast_pybaseball --post-metric-full-g
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import os

import pandas as pd


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _daterange_chunks(
    start: date, end: date, chunk_days: int
) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = start
    delta = timedelta(days=max(1, chunk_days))
    while cur <= end:
        chunk_end = min(cur + delta - timedelta(days=1), end)
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return out


def _shard_path(out_dir: Path, d0: date, d1: date) -> Path:
    return out_dir / "chunks" / f"statcast_{d0.isoformat()}__{d1.isoformat()}.parquet"


def _empty_marker_path(out_dir: Path, d0: date, d1: date) -> Path:
    return out_dir / "chunks" / f"statcast_{d0.isoformat()}__{d1.isoformat()}.empty"


def _import_statcast():
    try:
        from pybaseball import cache as pb_cache
        from pybaseball import statcast as pb_statcast
    except ImportError as e:
        print("Install pybaseball: pip install pybaseball", file=sys.stderr)
        raise SystemExit(1) from e
    return pb_cache, pb_statcast


def download_range(
    *,
    start: date,
    end: date,
    out_dir: Path,
    chunk_days: int,
    sleep_s: float,
    overwrite: bool,
    verbose: bool,
    parallel: bool,
) -> None:
    out_dir = out_dir.resolve()
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "pybaseball_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Must be set before pybaseball.cache singleton reads config (first import in process).
    os.environ["PYBASEBALL_CACHE"] = str(cache_dir)
    cache, statcast = _import_statcast()
    cache.enable()

    manifest: list[dict] = []
    for d0, d1 in _daterange_chunks(start, end, chunk_days):
        pq = _shard_path(out_dir, d0, d1)
        marker = _empty_marker_path(out_dir, d0, d1)
        if not overwrite and pq.is_file():
            manifest.append(
                {"start": d0.isoformat(), "end": d1.isoformat(), "rows": int(pd.read_parquet(pq).shape[0]), "path": str(pq)}
            )
            continue
        if not overwrite and marker.is_file():
            manifest.append(
                {"start": d0.isoformat(), "end": d1.isoformat(), "rows": 0, "path": str(marker)}
            )
            continue

        t0 = time.perf_counter()
        try:
            df = statcast(
                d0.isoformat(),
                d1.isoformat(),
                verbose=verbose,
                parallel=parallel,
            )
        except Exception as exc:
            print(f"ERROR {d0}..{d1}: {exc}", flush=True)
            raise

        elapsed = time.perf_counter() - t0
        n = 0 if df is None else len(df)
        if n > 0:
            if pq.exists() and overwrite:
                pq.unlink()
            df.to_parquet(pq, index=False)
            if marker.exists():
                marker.unlink()
            print(f"OK {d0}..{d1}  rows={n:,}  {elapsed:.1f}s  -> {pq.name}", flush=True)
            manifest.append(
                {"start": d0.isoformat(), "end": d1.isoformat(), "rows": n, "path": str(pq)}
            )
        else:
            marker.write_text("0\n", encoding="utf-8")
            print(f"EMPTY {d0}..{d1}  ({elapsed:.1f}s)", flush=True)
            manifest.append(
                {"start": d0.isoformat(), "end": d1.isoformat(), "rows": 0, "path": str(marker)}
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    (out_dir / "download_manifest.json").write_text(
        json.dumps({"start": start.isoformat(), "end": end.isoformat(), "chunks": manifest}, indent=2),
        encoding="utf-8",
    )


def combine_shards(out_dir: Path, combined_name: str = "statcast_all.parquet") -> None:
    out_dir = out_dir.resolve()
    chunks = sorted((out_dir / "chunks").glob("statcast_*.parquet"))
    if not chunks:
        raise SystemExit(f"No parquet shards under {out_dir / 'chunks'}")

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("combine needs pyarrow: pip install pyarrow") from e

    combined = out_dir / combined_name
    writer: pq.ParquetWriter | None = None
    try:
        for path in chunks:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(combined, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print(f"Wrote {combined} from {len(chunks)} shards.")


def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here.parent / "data" / "statcast_pybaseball"

    p = argparse.ArgumentParser(description="Download Statcast (pybaseball) in resumable chunks.")
    p.add_argument("--output-dir", type=Path, default=default_out, help="Root output directory.")
    p.add_argument("--start", default="2020-01-01", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive; default today (UTC date).")
    p.add_argument("--chunk-days", type=int, default=7, help="Length of each statcast window in days.")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep after each chunk (rate limit).")
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel sub-requests inside statcast.")
    p.add_argument("--quiet", action="store_true", help="Less console noise from pybaseball.")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch shards even if output exists.")
    p.add_argument("--combine-only", action="store_true", help="Only merge existing Parquet shards.")
    p.add_argument("--combined-name", default="statcast_all.parquet", help="Output filename for --combine-only.")
    p.add_argument(
        "--post-hitter-features",
        action="store_true",
        help=(
            "After a successful run, build hitter clustering features from the combined Parquet "
            "(requires statcast_hitter_feature_pipeline.py; output: statcast_hitter_features_2020_2026/). "
            "Use with --combine-only after shards exist, or ensure the combined file already exists."
        ),
    )
    p.add_argument(
        "--post-metric-full-g",
        action="store_true",
        help=(
            "After shard download completes, run Metric Folder/build_full_g_dataset.py on "
            "output-dir/chunks (BIP-filtered G dataset -> Full Data Set.parquet)."
        ),
    )
    p.add_argument(
        "--post-metric-rhb-only",
        action="store_true",
        help="With --post-metric-full-g, pass --rhb-only to the metric build (match RHH production).",
    )
    args = p.parse_args()

    out_dir: Path = args.output_dir
    if args.combine_only:
        combine_shards(out_dir, args.combined_name)
        if args.post_hitter_features:
            _run_hitter_feature_pipeline(out_dir / args.combined_name, here.parent)
        return

    start = _parse_date(args.start)
    if args.end:
        end = _parse_date(args.end)
    else:
        end = date.today()

    if end < start:
        raise SystemExit("--end must be on or after --start")

    print(f"Statcast download -> {out_dir.resolve()}", flush=True)
    print(f"Range {start} .. {end}  chunk_days={args.chunk_days}", flush=True)

    download_range(
        start=start,
        end=end,
        out_dir=out_dir,
        chunk_days=args.chunk_days,
        sleep_s=args.sleep,
        overwrite=args.overwrite,
        verbose=not args.quiet,
        parallel=not args.no_parallel,
    )
    print("Done. Manifest:", out_dir / "download_manifest.json", flush=True)

    if args.post_metric_full_g:
        _run_metric_full_g_build(here.parent, out_dir / "chunks", rhb_only=bool(args.post_metric_rhb_only))

    if args.post_hitter_features:
        combined = out_dir / args.combined_name
        if combined.is_file():
            _run_hitter_feature_pipeline(combined, here.parent)
        else:
            print(
                f"[post-hitter-features] No combined file at {combined}. "
                "Run: python scripts/download_statcast_pybaseball.py --combine-only --post-hitter-features "
                f"--output-dir {out_dir}",
                flush=True,
            )


def _run_metric_full_g_build(project_root: Path, chunks_dir: Path, *, rhb_only: bool) -> None:
    """Rebuild Metric Folder/Full Data Set.parquet from weekly Statcast shards."""
    script = project_root / "Metric Folder" / "build_full_g_dataset.py"
    if not script.is_file():
        print(f"[post-metric-full-g] Missing {script}", flush=True)
        return
    cmd = [
        sys.executable,
        str(script),
        "--statcast-chunks-dir",
        str(chunks_dir),
        "--require-download-manifest",
    ]
    if rhb_only:
        cmd.append("--rhb-only")
    print(f"[post-metric-full-g] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False, cwd=str(project_root))
    if proc.returncode != 0:
        print(f"[post-metric-full-g] build exited with code {proc.returncode}", flush=True)


def _run_hitter_feature_pipeline(combined_parquet: Path, project_root: Path) -> None:
    """Invoke statcast hitter feature pipeline on the data set from 2020 to 2026."""
    import importlib.util

    script = project_root / "scripts" / "statcast_hitter_feature_pipeline.py"
    spec = importlib.util.spec_from_file_location("statcast_hitter_feature_pipeline", script)
    if spec is None or spec.loader is None:
        print(f"[post-hitter-features] Could not load {script}", flush=True)
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = project_root / "statcast_hitter_features_2020_2026"
    out.mkdir(parents=True, exist_ok=True)
    single = out / "hitter_features_2020_2026.parquet"
    print(f"[post-hitter-features] Running pipeline on {combined_parquet} -> {single}", flush=True)
    mod.run_pipeline(input_path=combined_parquet, single_output=single)


if __name__ == "__main__":
    main()
