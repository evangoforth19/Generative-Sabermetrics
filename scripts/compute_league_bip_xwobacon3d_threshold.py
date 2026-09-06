#!/usr/bin/env python3
"""
League-wide BIP-only xwOBAcon_3D threshold from the local Statcast / pybaseball table
and the saved LightGBM value surface model.

tau_league_mean = mean(xwOBAcon_3D) over all valid league BIP (see markdown output).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "outputs" / "batted_ball_value_surface") not in sys.path:
    sys.path.insert(0, str(REPO / "outputs" / "batted_ball_value_surface"))

from batted_ball_value_model import HOME_X, HOME_Y, load_batted_ball_value_model, predict_xwobacon3d  # noqa: E402

HOME_PLATE_X = 125.42
HOME_PLATE_Y = 198.27
assert abs(HOME_X - HOME_PLATE_X) < 1e-6 and abs(HOME_Y - HOME_PLATE_Y) < 1e-6

WEIGHTS = np.array([0.0, 0.902, 1.279, 1.618, 2.078], dtype=np.float64)
CLASS_NAMES = ["out", "single", "double", "triple", "home_run"]
OUT_EVENTS = frozenset(
    {
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice_out",
        "sac_fly",
        "sac_fly_double_play",
        "triple_play",
    }
)
HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
EXCLUDE_EVENTS = frozenset(
    {
        "field_error",
        "fielders_choice",
        "catcher_interf",
        "hit_by_pitch",
        "walk",
        "strikeout",
        "strikeout_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "truncated_pa",
    }
)

PRED_BATCH = 100_000


def _discover_statcast_paths(repo: Path) -> list[Path]:
    """Heuristic search for local 2020–2026 Statcast / pybaseball tabular shards."""
    hits: list[Path] = []
    skip_parts = ("pybaseball_cache", ".venv", "node_modules", "__pycache__")
    for ext in (".parquet", ".csv", ".feather"):
        for p in repo.rglob(f"*{ext}"):
            if any(sp in p.parts for sp in skip_parts):
                continue
            s = str(p).lower()
            if not (("statcast" in s) or ("pybaseball" in s and "batted" in s)):
                continue
            if not any(y in s for y in ("2020", "2021", "2022", "2023", "2024", "2025", "2026")):
                continue
            hits.append(p.resolve())
    # Prefer league chunks directory when present
    chunks = repo / "data" / "statcast_pybaseball" / "chunks"
    if chunks.is_dir():
        shard = sorted(chunks.glob("statcast_*.parquet"))
        if shard:
            return shard
    return sorted(set(hits))


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suf in (".feather", ".fea"):
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table type: {path}")


def _resolve_data_paths(repo: Path, data_path: Path | None) -> list[Path]:
    if data_path is None:
        return _discover_statcast_paths(repo)
    p = data_path.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        return [p]
    if p.is_dir():
        files: list[Path] = []
        for pattern in ("*.parquet", "*.csv", "*.feather", "**/*.parquet", "**/*.csv", "**/*.feather"):
            files.extend(p.glob(pattern))
        files = sorted({f.resolve() for f in files if f.is_file()})
        if not files:
            raise FileNotFoundError(f"No tabular files under {p}")
        return files
    raise ValueError(f"--data-path must be file or directory: {p}")


def _spray_from_coords(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    x = pd.to_numeric(hc_x, errors="coerce")
    y = pd.to_numeric(hc_y, errors="coerce")
    return np.degrees(np.arctan2(x - HOME_PLATE_X, HOME_PLATE_Y - y))


def _bunt_mask(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    if "bb_type" in df.columns:
        m |= df["bb_type"].astype(str).str.lower().str.contains("bunt", na=False)
    for col in ("des", "description"):
        if col in df.columns:
            m |= df[col].astype(str).str.lower().str.contains("bunt", na=False)
    return m


def _map_outcome(events: Any) -> str | None:
    if events is None or (isinstance(events, float) and np.isnan(events)):
        return None
    e = str(events).strip().lower()
    if not e or e == "nan":
        return None
    if e in EXCLUDE_EVENTS:
        return None
    if e in HIT_EVENTS:
        return e
    if e in OUT_EVENTS:
        return "out"
    return None


def _extract_year(df: pd.DataFrame) -> pd.Series:
    if "game_year" in df.columns:
        y = pd.to_numeric(df["game_year"], errors="coerce")
        if y.notna().any():
            return y
    if "game_date" in df.columns:
        return pd.to_datetime(df["game_date"], errors="coerce").dt.year
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _prepare_shard(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return filtered BIP rows with EV, LA, SA, outcome_5, year; or None if unusable."""
    if "events" not in df.columns and "event" in df.columns:
        df = df.assign(events=df["event"])
    need = ("events", "launch_speed", "launch_angle")
    if not all(c in df.columns for c in need):
        return None
    if "type" in df.columns:
        m = df["type"].astype(str).str.upper().eq("X")
        df = df.loc[m].copy()
    else:
        df = df.copy()
    if len(df) == 0:
        return None

    ev = pd.to_numeric(df["launch_speed"], errors="coerce")
    la = pd.to_numeric(df["launch_angle"], errors="coerce")
    if "spray_angle_deg" in df.columns:
        sa = pd.to_numeric(df["spray_angle_deg"], errors="coerce")
    elif "hc_x" in df.columns and "hc_y" in df.columns:
        sa = _spray_from_coords(df["hc_x"], df["hc_y"])
    else:
        return None

    df["_EV"] = ev
    df["_LA"] = la
    df["_SA"] = sa
    df["_year"] = _extract_year(df)

    ok = df["_EV"].notna() & df["_LA"].notna() & df["_SA"].notna()
    df = df.loc[ok].copy()
    if len(df) == 0:
        return None

    df = df.loc[~_bunt_mask(df)].copy()
    if len(df) == 0:
        return None

    outcomes = df["events"].map(_map_outcome)
    df = df.loc[outcomes.notna()].copy()
    df["outcome_5"] = outcomes.loc[df.index]
    if len(df) == 0:
        return None

    return df[["_EV", "_LA", "_SA", "outcome_5", "_year"]].rename(
        columns={"_EV": "EV", "_LA": "LA", "_SA": "SA", "_year": "year"}
    )


def _realized_values(labels: np.ndarray) -> np.ndarray:
    lut = {"out": 0, "single": 1, "double": 2, "triple": 3, "home_run": 4}
    idx = pd.Series(labels.astype(str)).map(lut).to_numpy(dtype=np.intp)
    return WEIGHTS[idx]


def _percentiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct = {f"p{q:02d}_xwOBAcon_3D": float(np.percentile(x, q)) for q in qs if q != 50}
    pct["median_xwOBAcon_3D"] = float(np.percentile(x, 50))
    return pct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", type=Path, default=None, help="File or directory of Statcast tables.")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=REPO / "outputs" / "batted_ball_value_surface",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "outputs" / "league_bip_xwobacon3d_threshold",
    )
    ap.add_argument("--save-row-level", action="store_true")
    args = ap.parse_args()

    paths = _resolve_data_paths(REPO, args.data_path)
    model_dir = args.model_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_batted_ball_value_model(model_dir)

    n_loaded_total = 0
    xw_parts: list[np.ndarray] = []
    rw_parts: list[np.ndarray] = []
    yr_parts: list[np.ndarray] = []
    row_buf: list[pd.DataFrame] = []

    outcome_counts: dict[str, int] = {k: 0 for k in CLASS_NAMES}

    for path in paths:
        try:
            raw = _read_table(path)
        except Exception as ex:
            print(f"skip read fail {path}: {ex}", file=sys.stderr)
            continue
        n_loaded_total += len(raw)
        sub = _prepare_shard(raw)
        if sub is None or len(sub) == 0:
            continue
        for lab, c in sub["outcome_5"].value_counts().items():
            outcome_counts[str(lab)] = outcome_counts.get(str(lab), 0) + int(c)

        ev = sub["EV"].to_numpy(dtype=np.float64)
        la = sub["LA"].to_numpy(dtype=np.float64)
        sa = sub["SA"].to_numpy(dtype=np.float64)
        xw = np.empty(len(sub), dtype=np.float64)
        for s in range(0, len(sub), PRED_BATCH):
            e = min(s + PRED_BATCH, len(sub))
            xw[s:e] = predict_xwobacon3d(ev[s:e], la[s:e], sa[s:e], bundle, calibrated=True)
        rw = _realized_values(sub["outcome_5"].to_numpy())
        yr = pd.to_numeric(sub["year"], errors="coerce").to_numpy(dtype=np.float64)

        xw_parts.append(xw)
        rw_parts.append(rw)
        yr_parts.append(yr)
        if args.save_row_level:
            row_buf.append(
                pd.DataFrame(
                    {
                        "EV": ev,
                        "LA": la,
                        "SA": sa,
                        "xwOBAcon_3D": xw,
                        "realized_woba_value": rw,
                        "outcome_5": sub["outcome_5"].to_numpy(),
                        "year": yr,
                        "source_file": str(path.name),
                    }
                )
            )

    if not xw_parts:
        raise RuntimeError("No valid BIP rows after filtering; check --data-path and schema.")

    xw = np.concatenate(xw_parts)
    rw = np.concatenate(rw_parts)
    yr = np.concatenate(yr_parts)

    n_valid = int(len(xw))
    mean_x = float(np.mean(xw))
    std_x = float(np.std(xw, ddof=0))
    min_x = float(np.min(xw))
    max_x = float(np.max(xw))
    pct = _percentiles(xw)

    mean_rw = float(np.mean(rw))
    std_rw = float(np.std(rw, ddof=0))
    corr = float(np.corrcoef(xw, rw)[0, 1]) if n_valid > 1 else float("nan")

    summary = {
        "n_total_rows_loaded": int(n_loaded_total),
        "n_valid_bip": n_valid,
        "n_source_files": len(paths),
        "mean_xwOBAcon_3D": mean_x,
        "std_xwOBAcon_3D": std_x,
        "min_xwOBAcon_3D": min_x,
        **pct,
        "max_xwOBAcon_3D": max_x,
        "mean_realized_woba_value": mean_rw,
        "std_realized_woba_value": std_rw,
        "correlation_model_value_realized_value": corr,
    }

    by_year_rows: list[dict[str, Any]] = []
    if np.isfinite(yr).any():
        yint = np.where(np.isfinite(yr), yr, np.nan)
        for y in sorted({int(v) for v in yint[np.isfinite(yint)]}):
            m = yint == float(y)
            if not np.any(m):
                continue
            xv, rv = xw[m], rw[m]
            by_year_rows.append(
                {
                    "year": y,
                    "n_bip": int(np.sum(m)),
                    "mean_xwOBAcon_3D": float(np.mean(xv)),
                    "std_xwOBAcon_3D": float(np.std(xv, ddof=0)),
                    "median_xwOBAcon_3D": float(np.percentile(xv, 50)),
                    "p10_xwOBAcon_3D": float(np.percentile(xv, 10)),
                    "p25_xwOBAcon_3D": float(np.percentile(xv, 25)),
                    "p75_xwOBAcon_3D": float(np.percentile(xv, 75)),
                    "p90_xwOBAcon_3D": float(np.percentile(xv, 90)),
                    "mean_realized_woba_value": float(np.mean(rv)),
                }
            )

    if len(paths) == 1:
        dataset_path_str = str(paths[0])
    else:
        head = "; ".join(str(p) for p in paths[:3])
        dataset_path_str = head + (f"; ... (+{len(paths) - 3} more files)" if len(paths) > 3 else "")

    json_payload = {
        "tau_league_mean": mean_x,
        "tau_league_median": pct["median_xwOBAcon_3D"],
        "tau_league_p25": pct["p25_xwOBAcon_3D"],
        "tau_league_p10": pct["p10_xwOBAcon_3D"],
        "n_valid_bip": n_valid,
        "dataset_path": dataset_path_str,
        "n_source_files": len(paths),
        "model_dir": str(model_dir),
        "outcome_class_counts": outcome_counts,
        "weights_order": CLASS_NAMES,
        "weights_values": WEIGHTS.tolist(),
    }
    (out_dir / "league_bip_xwobacon3d_threshold.json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([summary]).to_csv(out_dir / "league_bip_xwobacon3d_summary.csv", index=False)
    if by_year_rows:
        pd.DataFrame(by_year_rows).to_csv(out_dir / "league_bip_xwobacon3d_by_year.csv", index=False)
    else:
        (out_dir / "league_bip_xwobacon3d_by_year.csv").write_text("year,n_bip\n", encoding="utf-8")

    skew_hint = (
        "right-skewed (mass at lower xwOBAcon with a long upper tail)"
        if mean_x > pct["median_xwOBAcon_3D"] + 0.02
        else "roughly symmetric or mildly skewed"
        if abs(mean_x - pct["median_xwOBAcon_3D"]) <= 0.02
        else "left-skewed (mean below median)"
    )
    md_lines = [
        "# League BIP xwOBAcon_3D threshold",
        "",
        "This is a **balls-in-play (BIP) only** summary: Statcast `type == X` rows with valid "
        "`launch_speed`, `launch_angle`, and spray angle (from `spray_angle_deg` or `hc_x`/`hc_y`). "
        "It is **not** a full plate-appearance xwOBA (walks, strikeouts, HBP, etc. are excluded).",
        "",
        "## Model and threshold",
        "",
        "The saved value surface model in `outputs/batted_ball_value_surface/` maps "
        "(EV, LA, SA) to calibrated **5-class** outcome probabilities "
        "`P(Y | EV, LA, SA)` for `Y ∈ {out, single, double, triple, home_run}`.",
        "",
        "Define **xwOBAcon_3D** as the linear expectation using Statcast-style linear weights on outcomes:",
        "",
        "```",
        "xwOBAcon_3D = sum_k P(class_k) * w_k",
        "```",
        "",
        "| class | weight |",
        "|---|---:|",
        "| out | 0.000 |",
        "| single | 0.902 |",
        "| double | 1.279 |",
        "| triple | 1.618 |",
        "| home_run | 2.078 |",
        "",
        "For downstream **Omega / downside-risk** style cutoffs, the primary pooled threshold is:",
        "",
        f"- **`tau_league_mean`** = **{mean_x:.6f}** = mean xwOBAcon_3D over all valid league BIP in this run.",
        f"- **`tau_league_median`** = **{pct['median_xwOBAcon_3D']:.6f}**",
        f"- **`tau_league_p25`** = **{pct['p25_xwOBAcon_3D']:.6f}**",
        f"- **`tau_league_p10`** = **{pct['p10_xwOBAcon_3D']:.6f}**",
        "",
        "## Pooled summary",
        "",
        f"- Rows scanned (all loaded tables): **{n_loaded_total:,}**",
        f"- Valid BIP after filters: **{n_valid:,}**",
        f"- Source files: **{len(paths)}**",
        f"- Mean / std (xwOBAcon_3D): **{mean_x:.6f}** / **{std_x:.6f}**",
        f"- Realized linear weight mean / std: **{mean_rw:.6f}** / **{std_rw:.6f}**",
        f"- Corr(xwOBAcon_3D, realized): **{corr:.6f}**",
        "",
        "### Outcome class counts (filtered BIP)",
        "",
        "```json",
        json.dumps(outcome_counts, indent=2),
        "```",
        "",
        "### Distribution shape (pooled)",
        "",
        f"- Min / max: **{min_x:.6f}** / **{max_x:.6f}**",
        f"- Interpretation: pooled xwOBAcon_3D looks **{skew_hint}** comparing mean vs median.",
        "",
        "## By-year summary",
        "",
    ]
    if by_year_rows:
        by_df = pd.DataFrame(by_year_rows)
        hdr = "| " + " | ".join(by_df.columns) + " |"
        sep = "|" + "|".join(["---"] * len(by_df.columns)) + "|"
        body = "\n".join(
            "| " + " | ".join(str(by_df.iloc[i][c]) for c in by_df.columns) + " |" for i in range(len(by_df))
        )
        md_lines.extend([hdr, sep, body])
    else:
        md_lines.append("_No `game_year` / parseable `game_date` in inputs; by-year table not produced._")
    md_lines.extend(["", "## Artifacts", "", "- `league_bip_xwobacon3d_threshold.json`", "- `league_bip_xwobacon3d_summary.csv`", "- `league_bip_xwobacon3d_by_year.csv`", ""])
    (out_dir / "league_bip_xwobacon3d_threshold.md").write_text("\n".join(md_lines), encoding="utf-8")

    if args.save_row_level and row_buf:
        pd.concat(row_buf, ignore_index=True).to_parquet(
            out_dir / "league_bip_xwobacon3d_row_level.parquet",
            index=False,
        )

    json_path = out_dir / "league_bip_xwobacon3d_threshold.json"
    md_path = out_dir / "league_bip_xwobacon3d_threshold.md"
    print(out_dir)
    print(json_path)
    print(md_path)
    print(n_valid)
    print(mean_x)
    print(pct["median_xwOBAcon_3D"])
    print(pct["p25_xwOBAcon_3D"])
    print(pct["p10_xwOBAcon_3D"])


if __name__ == "__main__":
    main()
