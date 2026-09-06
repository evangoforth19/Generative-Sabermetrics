"""Build G_COLUMNS extract from a Statcast pickle or from pybaseball weekly Parquet shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SBI_ROOT = ROOT / "MCMC 2" / "sbi_forward_sim"
sys.path.insert(0, str(SBI_ROOT))

from src.feature_engineering import add_spin_axis_trig, z_count_from_balls_strikes  # noqa: E402
from src.schema import G_COLUMNS  # noqa: E402

DEFAULT_PICKLE = ROOT / "Manifold Construction" / "batter_data_2024_2025.pkl"
DEFAULT_STATCAST_CHUNKS = ROOT / "data" / "statcast_pybaseball" / "chunks"
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_PARQUET = OUT_DIR / "Full Data Set.parquet"
DEFAULT_LINEAGE = OUT_DIR / "Full_Data_Set_column_lineage.json"

# Align with project BIP filters (Statcast launch_speed in mph).
MIN_LAUNCH_SPEED_MPH = 40.0


def _bip_collision_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that look like balls in play / contact (Statcast launch_speed present and plausible)."""
    if "launch_speed" not in df.columns:
        return pd.Series(False, index=df.index)
    ls = pd.to_numeric(df["launch_speed"], errors="coerce")
    return ls.notna() & (ls >= MIN_LAUNCH_SPEED_MPH)


def _transform_block(df: pd.DataFrame) -> pd.DataFrame:
    out = add_spin_axis_trig(df, axis_col="spin_axis")
    out = z_count_from_balls_strikes(out, table_name="statcast_batter_pitches")
    missing = [c for c in G_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing G_COLUMNS after transforms: {missing}")
    extra = ["spin_axis", "balls", "strikes"]
    for c in extra:
        if c not in out.columns:
            raise ValueError(f"Missing source column {c!r}")
    final_cols = list(G_COLUMNS) + extra
    return out[final_cols].copy()


def build_from_pickle(pickle_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(pickle_path)
    return _transform_block(df)


def build_from_statcast_chunks(
    chunks_dir: Path,
    *,
    rhb_only: bool,
) -> pd.DataFrame:
    paths = sorted(chunks_dir.glob("statcast_*.parquet"))
    if not paths:
        raise SystemExit(f"No statcast_*.parquet under {chunks_dir}")

    pieces: list[pd.DataFrame] = []
    buf: list[pd.DataFrame] = []
    total_rows = 0

    def flush_buf() -> None:
        nonlocal buf, pieces, total_rows
        if not buf:
            return
        cat = pd.concat(buf, ignore_index=True)
        buf = []
        pieces.append(cat)
        total_rows += len(cat)

    for path in paths:
        df = pd.read_parquet(path)
        if rhb_only and "stand" in df.columns:
            st = df["stand"].astype(str).str.upper().str.strip()
            df = df[st.str.startswith("R")].copy()
        df = df.loc[_bip_collision_mask(df)].copy()
        if df.empty:
            continue
        buf.append(_transform_block(df))
        # Limit peak RAM: flush every few shards.
        if len(buf) >= 8:
            flush_buf()

    flush_buf()
    if not pieces:
        raise SystemExit("No in-play rows (launch_speed) found across shards; nothing to write.")
    out = pd.concat(pieces, ignore_index=True)
    return out


def _write_lineage(
    path: Path,
    *,
    source: str,
    extra_notes: dict,
) -> None:
    lineage = {
        "source": source,
        "bip_filter": f"launch_speed >= {MIN_LAUNCH_SPEED_MPH} (Statcast mph)",
        "g_columns_order": list(G_COLUMNS),
        "source_columns_appended": {
            "spin_axis": (
                "Statcast spin axis in degrees; source for spin_axis_sin = sin(rad), "
                "spin_axis_cos = cos(rad) with rad = deg2rad(spin_axis)."
            ),
            "balls": (
                "Statcast balls; used with strikes to map z_count via "
                "feature_engineering.z_count_from_balls_strikes."
            ),
            "strikes": "Statcast strikes; used with balls to map z_count.",
        },
        "direct_from_statcast_no_transform": [
            "release_speed",
            "release_spin_rate",
            "plate_x",
            "plate_z",
            "pitch_type",
            "stand",
            "p_throws",
            "vx0",
            "vy0",
            "vz0",
            "ax",
            "ay",
            "az",
        ],
        **extra_notes,
    }
    path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build Full Data Set.parquet (G + spin_axis, balls, strikes).")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--pickle",
        type=Path,
        default=None,
        help=f"Statcast/batter pickle (default used if neither source flag is set): {DEFAULT_PICKLE}",
    )
    src.add_argument(
        "--statcast-chunks-dir",
        type=Path,
        default=None,
        help=(
            "Directory of statcast_*.parquet weekly shards (pybaseball download). "
            f"Typical: {DEFAULT_STATCAST_CHUNKS}."
        ),
    )
    p.add_argument(
        "--require-download-manifest",
        action="store_true",
        help="Require download_manifest.json next to the chunks directory (full pybaseball shard run finished).",
    )
    p.add_argument(
        "--rhb-only",
        action="store_true",
        help="When using --statcast-chunks-dir, keep only stand starting with R (match RHH production filters).",
    )
    p.add_argument("--output-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    p.add_argument("--output-lineage", type=Path, default=DEFAULT_LINEAGE)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.statcast_chunks_dir is not None:
        chunks_dir = args.statcast_chunks_dir.resolve()
        manifest = chunks_dir.parent / "download_manifest.json"
        if args.require_download_manifest and not manifest.is_file():
            raise SystemExit(f"Missing {manifest}; Statcast download not finished.")
        if not args.require_download_manifest and not manifest.is_file():
            print(
                f"WARNING: {manifest} not found — building from partial shards in {chunks_dir}.",
                flush=True,
            )
        out = build_from_statcast_chunks(chunks_dir, rhb_only=args.rhb_only)
        source_note = {"statcast_chunks_dir": str(chunks_dir), "rhb_only": bool(args.rhb_only)}
        _write_lineage(args.output_lineage, source="statcast_pybaseball_shards", extra_notes=source_note)
    else:
        pickle_path = (args.pickle or DEFAULT_PICKLE).resolve()
        if not pickle_path.is_file():
            raise SystemExit(f"Pickle not found: {pickle_path}")
        out = build_from_pickle(pickle_path)
        _write_lineage(
            args.output_lineage,
            source="pickle",
            extra_notes={"pickle_path": str(pickle_path)},
        )

    out.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_parquet} shape={out.shape}", flush=True)
    print(f"Wrote {args.output_lineage}", flush=True)


if __name__ == "__main__":
    main()
