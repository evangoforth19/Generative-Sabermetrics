#!/usr/bin/env python3
"""
Rank 12 simulation hitters (2021–2025) by wRC+ (FanGraphs / pybaseball),
xwOBAcon, and xwOBA (Baseball Savant custom leaderboard).

Does not include 2026. No plots.
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_START = 2021
SEASON_END = 2025

TARGET_PLAYERS: list[str] = [
    "Aaron Judge",
    "Alex Bregman",
    "Evan Longoria",
    "George Springer",
    "Giancarlo Stanton",
    "Kris Bryant",
    "Luis Robert",
    "Manny Machado",
    "Mike Trout",
    "Mookie Betts",
    "Nolan Arenado",
    "Pete Alonso",
]

OUTPUT_DIR = Path("outputs/simulation_hitter_public_metric_rankings_2021_2025")

# Populated during run for markdown warnings
_WARNINGS: list[str] = []
_FG_WRC_METHOD: str = ""
_SAVANT_XWOBACON_WEIGHTING: str = ""  # "BBE-weighted" or "PA-weighted (BBE unavailable)"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def savant_to_standard_name(name: str) -> str:
    """Convert 'Last, First' Savant style to 'First Last'."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    name = str(name).strip()
    if not name:
        return ""
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return " ".join(name.split())


def normalize_name(name: str) -> str:
    """
    Canonical lowercase key for matching FanGraphs / Savant / display names.
    Strips common suffixes (Jr., Sr., etc.) so 'Luis Robert' matches 'Robert Jr., Luis'.
    """
    s = savant_to_standard_name(name)
    s = " ".join(s.lower().split())
    for suf in (" jr.", " sr.", " iii", " ii", " iv", " v."):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    return s


def target_key_set() -> set[str]:
    return {normalize_name(p) for p in TARGET_PLAYERS}


def display_name_for_key(key: str) -> str:
    """Map normalized key back to the canonical display string from TARGET_PLAYERS."""
    for p in TARGET_PLAYERS:
        if normalize_name(p) == key:
            return p
    return key.title()


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return first actual column name matching candidates case-insensitively."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def find_player_name_column(df: pd.DataFrame) -> Optional[str]:
    """Savant CSV often uses ``last_name, first_name`` as a single header."""
    c = find_col(df, ("player_name", "player", "name"))
    if c:
        return c
    for col in df.columns:
        s = str(col).lower().replace(" ", "")
        if "last_name" in s and "first_name" in s:
            return str(col)
    return None


def safe_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean; ignores pairs where value or weight is NaN; returns NaN if no valid mass."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    w = pd.to_numeric(weights, errors="coerce").astype(float)
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _session_get(url: str, timeout: int = 60) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            return requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            time.sleep(0.8 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _fg_needs_pa_weight(fg: pd.DataFrame, keys: set[str]) -> bool:
    """True if any targeted player has more than one row (season-level split)."""
    name_col = find_col(fg, ("Name", "Player"))
    if not name_col:
        return True
    sub = fg.copy()
    sub["_norm"] = sub[name_col].map(normalize_name)
    sub = sub[sub["_norm"].isin(keys)]
    if sub.empty:
        return False
    mx = sub.groupby("_norm").size().max()
    return bool(mx > 1)


def load_fangraphs_wrc_plus() -> tuple[pd.DataFrame, str]:
    """
    Returns DataFrame columns: norm_key, player, PA, wRC_plus_2021_2025, fg_rows_per_player

    Uses ``batting_stats(2021, 2025, qual=0)`` first. If FanGraphs returns season-level rows,
    retries ``split_seasons=False`` for a true multi-year aggregate when available; otherwise
    PA-weights wRC+ across seasons.
    """
    global _FG_WRC_METHOD
    try:
        from pybaseball import batting_stats
    except ImportError as e:
        raise SystemExit("Install pybaseball: pip install pybaseball") from e

    keys = target_key_set()
    out_rows: list[dict] = []

    def _process_fg(fg: pd.DataFrame, source_label: str) -> bool:
        nonlocal out_rows
        name_col = find_col(fg, ("Name", "Player"))
        pa_col = find_col(fg, ("PA",))
        wrc_col = find_col(fg, ("wRC+", "wrc+", "WRC+"))
        if not name_col or not pa_col or not wrc_col:
            return False
        sub = fg.copy()
        sub["_norm"] = sub[name_col].map(normalize_name)
        sub = sub[sub["_norm"].isin(keys)]
        if sub.empty:
            return False
        grouped = sub.groupby("_norm", dropna=False)
        rows_per = grouped.size()
        use_direct = bool((rows_per == 1).all())
        for nk, chunk in grouped:
            pa = pd.to_numeric(chunk[pa_col], errors="coerce")
            wrc = pd.to_numeric(chunk[wrc_col], errors="coerce")
            total_pa = float(pa.sum())
            if use_direct:
                wrc_val = float(wrc.iloc[0]) if len(wrc) else float("nan")
            else:
                wrc_val = safe_weighted_average(wrc, pa)
            disp = display_name_for_key(str(nk))
            out_rows.append(
                {
                    "norm_key": nk,
                    "player": disp,
                    "PA": total_pa,
                    "wRC_plus_2021_2025": wrc_val,
                    "fg_rows_per_player": int(len(chunk)),
                }
            )
        global _FG_WRC_METHOD
        if use_direct and "split_seasons=False" in source_label:
            _FG_WRC_METHOD = (
                "FanGraphs multi-season aggregate row (pybaseball batting_stats, "
                "split_seasons=False, qual=0)"
            )
        elif use_direct:
            _FG_WRC_METHOD = (
                "FanGraphs: single row per player from pybaseball batting_stats(2021, 2025, qual=0)"
            )
        else:
            _FG_WRC_METHOD = (
                "FanGraphs: PA-weighted average of wRC+ across season-level rows "
                f"({source_label}, qual=0)"
            )
        return True

    try:
        fg_primary = batting_stats(SEASON_START, SEASON_END, qual=0)
    except Exception as e:
        _FG_WRC_METHOD = (
            f"FanGraphs: batting_stats(2021, 2025, qual=0) failed ({type(e).__name__}: {e})"
        )
        return pd.DataFrame(), _FG_WRC_METHOD

    if not fg_primary.empty and _fg_needs_pa_weight(fg_primary, keys):
        try:
            fg_agg = batting_stats(SEASON_START, SEASON_END, qual=0, split_seasons=False)
            if _process_fg(fg_agg, "split_seasons=False"):
                return pd.DataFrame(out_rows), _FG_WRC_METHOD
        except Exception:
            pass
        out_rows.clear()

    if not _process_fg(fg_primary, "batting_stats(2021, 2025, qual=0)"):
        _FG_WRC_METHOD = "FanGraphs: failed to resolve Name/PA/wRC+ columns or no matching players"
    return pd.DataFrame(out_rows), _FG_WRC_METHOD


def _savant_url(year: int, *, with_batted_ball: bool, min_param: str) -> str:
    sel = "pa,xwoba,wobacon,xwobacon"
    if with_batted_ball:
        sel += ",batted_ball"
    sel_q = quote(sel, safe="")
    return (
        "https://baseballsavant.mlb.com/leaderboard/custom?"
        f"year={year}&type=batter&filter=&min={min_param}&selections={sel_q}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&sort=xwobacon&sortDir=desc&csv=true"
    )


def _savant_url_html(year: int, *, with_batted_ball: bool, min_param: str) -> str:
    sel = "pa,xwoba,wobacon,xwobacon"
    if with_batted_ball:
        sel += ",batted_ball"
    sel_q = quote(sel, safe="")
    return (
        "https://baseballsavant.mlb.com/leaderboard/custom?"
        f"year={year}&type=batter&filter=&min={min_param}&selections={sel_q}"
        "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&sort=xwobacon&sortDir=desc"
    )


def fetch_savant_custom_leaderboard(year: int) -> pd.DataFrame:
    """
    Download Savant custom batter leaderboard for one season.
    Tries CSV URL variants; falls back to read_html on the HTML leaderboard page.
    """
    attempts: list[str] = []
    last_err: Optional[Exception] = None

    def _read_csv_from_text(text: str) -> pd.DataFrame:
        text = text.lstrip("\ufeff")
        lines = text.splitlines()
        if not lines:
            raise ValueError("empty CSV body")
        # Strip BOM / leading junk; find header row with Savant-ish columns
        start = 0
        for i, line in enumerate(lines):
            low = line.lower()
            if ("xwoba" in low or "xwobacon" in low) and ("pa" in low or "player" in low):
                if "," in line or "\t" in line:
                    start = i
                    break
        body = "\n".join(lines[start:])
        sep = "\t" if "\t" in lines[start] else ","
        return pd.read_csv(io.StringIO(body), sep=sep, low_memory=False)

    def _read_html_tables(html: str) -> list[pd.DataFrame]:
        errs: list[Exception] = []
        for flavor in ("lxml", "bs4", "html5lib"):
            try:
                return pd.read_html(io.StringIO(html), flavor=flavor)
            except Exception as e:
                errs.append(e)
        raise RuntimeError(
            "pandas.read_html failed for all flavors (lxml, bs4, html5lib): "
            + "; ".join(repr(e) for e in errs)
        )

    for with_bb in (True, False):
        for min_param in ("0", "q"):
            url = _savant_url(year, with_batted_ball=with_bb, min_param=min_param)
            attempts.append(url)
            try:
                r = _session_get(url)
                r.raise_for_status()
                head = r.text[:800].lower()
                if "<html" in head or "<!doctype" in head:
                    raise ValueError("expected CSV but received HTML document")
                df = _read_csv_from_text(r.text)
                if len(df.columns) > 1 and find_player_name_column(df):
                    return df
            except Exception as e:
                last_err = e

    for with_bb in (True, False):
        for min_param in ("0", "q"):
            url = _savant_url_html(year, with_batted_ball=with_bb, min_param=min_param)
            attempts.append(f"{url} (read_html)")
            try:
                r = _session_get(url)
                r.raise_for_status()
                tables = _read_html_tables(r.text)
                for tbl in tables:
                    if find_player_name_column(tbl) and find_col(tbl, ("pa",)):
                        return tbl
            except Exception as e:
                last_err = e

    print(
        "ERROR: Baseball Savant fetch failed for year "
        f"{year}. Last error: {last_err!r}\nURLs attempted:\n"
        + "\n".join(f"  - {u}" for u in attempts),
        file=sys.stderr,
    )
    raise RuntimeError(f"Savant leaderboard fetch failed for {year}") from last_err


def aggregate_savant_metrics() -> tuple[pd.DataFrame, str]:
    """
    Per player 2021–2025: PA-weighted xwOBA; xwOBAcon weighted by BBE if column exists else PA.
    Returns (df, xwobacon_weight_note).
    """
    global _SAVANT_XWOBACON_WEIGHTING
    keys = target_key_set()
    yearly: list[pd.DataFrame] = []
    has_bbe_any_year = False
    any_year_had_bbe_col = False

    for y in range(SEASON_START, SEASON_END + 1):
        raw = fetch_savant_custom_leaderboard(y)
        name_c = find_player_name_column(raw)
        first_c = find_col(raw, ("first_name", "firstname"))
        last_c = find_col(raw, ("last_name", "lastname"))
        if name_c:
            name_series = raw[name_c].astype(str)
        elif first_c and last_c:
            name_series = raw[last_c].astype(str) + ", " + raw[first_c].astype(str)
        else:
            raise RuntimeError(
                f"Savant {y}: could not resolve player name column "
                "(expected player_name/name, last_name+first_name, or combined Savant header)"
            )
        pa_c = find_col(raw, ("pa",))
        xwoba_c = find_col(raw, ("xwoba", "xwOBA", "xWOBA"))
        xwobacon_c = find_col(raw, ("xwobacon", "xwOBAcon", "x_wobacon"))
        bbe_c = find_col(
            raw,
            ("batted_ball", "batted_balls", "bbe", "BBE", "balls_in_play"),
        )
        if not all([pa_c, xwoba_c, xwobacon_c]):
            raise RuntimeError(f"Savant {y}: missing required columns after detection")

        part = pd.DataFrame(
            {
                "player_raw": name_series.astype(str),
                "norm_key": name_series.map(normalize_name),
                "PA": pd.to_numeric(raw[pa_c], errors="coerce"),
                "xwOBA": pd.to_numeric(raw[xwoba_c], errors="coerce"),
                "xwOBAcon": pd.to_numeric(raw[xwobacon_c], errors="coerce"),
                "year": y,
            }
        )
        if bbe_c:
            any_year_had_bbe_col = True
            part["BBE"] = pd.to_numeric(raw[bbe_c], errors="coerce")
            has_bbe_any_year = has_bbe_any_year or bool(part["BBE"].notna().any())
        else:
            part["BBE"] = np.nan
        part = part[part["norm_key"].isin(keys)]
        yearly.append(part)

    all_y = pd.concat(yearly, ignore_index=True)
    use_bbe_weight = bool(
        any_year_had_bbe_col and has_bbe_any_year and all_y["BBE"].notna().any()
    )
    if use_bbe_weight:
        _SAVANT_XWOBACON_WEIGHTING = (
            "Per-season xwOBAcon combined with BBE weights where BBE is present and "
            "> 0; otherwise PA weight for that season (Baseball Savant custom leaderboard)"
        )
    else:
        _SAVANT_XWOBACON_WEIGHTING = (
            "PA-weighted yearly xwOBAcon (BBE column unavailable or all missing; "
            "Baseball Savant custom leaderboard)"
        )

    rows: list[dict] = []
    for nk in sorted(keys):
        chunk = all_y[all_y["norm_key"] == nk]
        if chunk.empty:
            rows.append(
                {
                    "norm_key": nk,
                    "player": display_name_for_key(nk),
                    "PA_savant": float("nan"),
                    "BBE": float("nan"),
                    "xwOBA_2021_2025": float("nan"),
                    "xwOBAcon_2021_2025": float("nan"),
                }
            )
            continue
        total_pa = float(chunk["PA"].sum())
        total_bbe = float(chunk["BBE"].sum()) if chunk["BBE"].notna().any() else float("nan")
        xwoba_agg = safe_weighted_average(chunk["xwOBA"], chunk["PA"])
        if use_bbe_weight:
            bbe_ok = chunk["BBE"].notna() & (chunk["BBE"] > 0)
            w_xwobacon = np.where(bbe_ok, chunk["BBE"], chunk["PA"])
            w_xwobacon = pd.Series(w_xwobacon, index=chunk.index)
        else:
            w_xwobacon = chunk["PA"]
        xwobacon_agg = safe_weighted_average(chunk["xwOBAcon"], w_xwobacon)
        rows.append(
            {
                "norm_key": nk,
                "player": display_name_for_key(nk),
                "PA_savant": total_pa,
                "BBE": total_bbe if not np.isnan(total_bbe) else float("nan"),
                "xwOBA_2021_2025": xwoba_agg,
                "xwOBAcon_2021_2025": xwobacon_agg,
            }
        )
    return pd.DataFrame(rows), _SAVANT_XWOBACON_WEIGHTING


def merge_and_rank(
    fg_df: pd.DataFrame, savant_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Merge sources, compute ranks (higher metric = better = rank 1), average_rank."""
    warnings_local: list[str] = []
    keys = [normalize_name(p) for p in TARGET_PLAYERS]

    if fg_df.empty or "norm_key" not in fg_df.columns:
        fg_by = pd.DataFrame(index=pd.Index([], name="norm_key"))
        warnings_local.append(
            "FanGraphs: no player table loaded (fetch failure, 403/block, or no matching names)."
        )
    else:
        fg_by = fg_df.set_index("norm_key")
    if savant_df.empty or "norm_key" not in savant_df.columns:
        sv_by = pd.DataFrame(index=pd.Index([], name="norm_key"))
        warnings_local.append(
            "Baseball Savant: no player table loaded (network/DNS failure or parse error)."
        )
    else:
        sv_by = savant_df.set_index("norm_key")

    merged_rows: list[dict] = []
    for nk in keys:
        fr = fg_by.loc[nk] if nk in fg_by.index else None
        sr = sv_by.loc[nk] if nk in sv_by.index else None

        if fr is None or (isinstance(fr, pd.DataFrame) and fr.empty):
            pa_fg = float("nan")
            wrc = float("nan")
            if not fg_df.empty:
                warnings_local.append(f"FanGraphs: no row for {display_name_for_key(nk)}")
        else:
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[0]
            pa_fg = float(fr["PA"])
            wrc = float(fr["wRC_plus_2021_2025"])

        if sr is None or (isinstance(sr, pd.DataFrame) and sr.empty):
            pa_sv = float("nan")
            bbe = float("nan")
            xw = float("nan")
            xwc = float("nan")
            if not savant_df.empty:
                warnings_local.append(f"Baseball Savant: no row for {display_name_for_key(nk)}")
        else:
            if isinstance(sr, pd.DataFrame):
                sr = sr.iloc[0]
            pa_sv = float(sr["PA_savant"])
            bbe = float(sr["BBE"])
            xw = float(sr["xwOBA_2021_2025"])
            xwc = float(sr["xwOBAcon_2021_2025"])

        pa_out = pa_fg if pd.notna(pa_fg) and pa_fg > 0 else pa_sv

        merged_rows.append(
            {
                "norm_key": nk,
                "player": display_name_for_key(nk),
                "PA": pa_out,
                "BBE": bbe,
                "wRC_plus_2021_2025": wrc,
                "xwOBAcon_2021_2025": xwc,
                "xwOBA_2021_2025": xw,
            }
        )

    out = pd.DataFrame(merged_rows)

    out["rank_wRC_plus"] = out["wRC_plus_2021_2025"].rank(
        ascending=False, method="min"
    )
    out["rank_xwOBAcon"] = out["xwOBAcon_2021_2025"].rank(
        ascending=False, method="min"
    )
    out["rank_xwOBA"] = out["xwOBA_2021_2025"].rank(ascending=False, method="min")
    out["average_rank"] = out[["rank_wRC_plus", "rank_xwOBAcon", "rank_xwOBA"]].mean(
        axis=1, skipna=False
    )
    out = out.sort_values("average_rank", ascending=True, na_position="last")
    return out, list(dict.fromkeys(warnings_local))


def write_markdown_summary(
    path: Path,
    combined: pd.DataFrame,
    extra_warnings: list[str],
    fg_method: str,
    savant_xwobacon_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# Public metric rankings (2021–2025)\n")
    lines.append("## Seasons\n")
    lines.append(
        f"- **Seasons used:** {SEASON_START}–{SEASON_END} (five completed MLB seasons; 2026 excluded)\n"
    )
    lines.append("## Data sources\n")
    lines.append("- **wRC+:** FanGraphs via `pybaseball.batting_stats`\n")
    lines.append("- **xwOBA, xwOBAcon:** Baseball Savant custom batter leaderboard (CSV / HTML fallback)\n")
    lines.append("## Weighting\n")
    lines.append(f"- **wRC+:** {fg_method}\n")
    lines.append("- **xwOBA:** PA-weighted average of single-season xwOBA over 2021–2025\n")
    lines.append(f"- **xwOBAcon:** {savant_xwobacon_note}\n")
    lines.append("\n## Final combined ranking (sorted by average_rank)\n")
    lines.append(_df_to_markdown(combined))
    lines.append("\n\n## By wRC+ (descending)\n")
    by_wrc = combined.sort_values("wRC_plus_2021_2025", ascending=False, na_position="last")
    lines.append(_df_to_markdown(by_wrc))
    lines.append("\n\n## By xwOBAcon (descending)\n")
    by_xwc = combined.sort_values("xwOBAcon_2021_2025", ascending=False, na_position="last")
    lines.append(_df_to_markdown(by_xwc))
    lines.append("\n\n## By xwOBA (descending)\n")
    by_xw = combined.sort_values("xwOBA_2021_2025", ascending=False, na_position="last")
    lines.append(_df_to_markdown(by_xw))

    all_warn = list(dict.fromkeys(_WARNINGS + extra_warnings))
    if all_warn:
        lines.append("\n\n## Missing-data and fetch warnings\n")
        for w in all_warn:
            lines.append(f"- {w}\n")

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        fg_df, fg_method = load_fangraphs_wrc_plus()
    except Exception as e:
        _WARNINGS.append(f"FanGraphs / pybaseball failed: {e!r}")
        fg_df = pd.DataFrame()
        fg_method = f"FAILED: {e!r}"

    try:
        savant_df, savant_note = aggregate_savant_metrics()
    except Exception as e:
        _WARNINGS.append(f"Baseball Savant aggregation failed: {e!r}")
        savant_df = pd.DataFrame(
            columns=[
                "norm_key",
                "player",
                "PA_savant",
                "BBE",
                "xwOBA_2021_2025",
                "xwOBAcon_2021_2025",
            ]
        )
        savant_note = "PA-weighted (fetch failed)"

    combined, merge_warn = merge_and_rank(fg_df, savant_df)

    combined_path = OUTPUT_DIR / "combined_hitter_metrics_2021_2025.csv"
    combined_out = combined[
        [
            "player",
            "PA",
            "BBE",
            "wRC_plus_2021_2025",
            "xwOBAcon_2021_2025",
            "xwOBA_2021_2025",
            "rank_wRC_plus",
            "rank_xwOBAcon",
            "rank_xwOBA",
            "average_rank",
        ]
    ]
    combined_out.to_csv(combined_path, index=False)

    rank_wrc = combined_out.sort_values("wRC_plus_2021_2025", ascending=False, na_position="last")
    rank_wrc.to_csv(OUTPUT_DIR / "rank_by_wrc_plus.csv", index=False)
    rank_xwc = combined_out.sort_values("xwOBAcon_2021_2025", ascending=False, na_position="last")
    rank_xwc.to_csv(OUTPUT_DIR / "rank_by_xwobacon.csv", index=False)
    rank_xw = combined_out.sort_values("xwOBA_2021_2025", ascending=False, na_position="last")
    rank_xw.to_csv(OUTPUT_DIR / "rank_by_xwoba.csv", index=False)

    md_path = OUTPUT_DIR / "public_metric_ranking_summary.md"
    write_markdown_summary(
        md_path,
        combined_out,
        merge_warn,
        fg_method,
        savant_note,
    )

    # Terminal output (only paths + table as specified)
    print(str(OUTPUT_DIR.resolve()))
    print(str(combined_path.resolve()))
    print(str(md_path.resolve()))
    cols = [
        "player",
        "wRC_plus_2021_2025",
        "xwOBAcon_2021_2025",
        "xwOBA_2021_2025",
        "rank_wRC_plus",
        "rank_xwOBAcon",
        "rank_xwOBA",
        "average_rank",
    ]
    print(combined_out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
