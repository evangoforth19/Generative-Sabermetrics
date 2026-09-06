#!/usr/bin/env python3
"""
Statcast-only hitter feature engineering for clustering (PCA / GMM).

Builds hitter-level aggregates from the **data set from 2020 to 2026** (pitch-by-pitch
Statcast-style table), saves CSV artifacts under ``statcast_hitter_features_2020_2026``.

Run:
  python scripts/statcast_hitter_feature_pipeline.py --input path/to/statcast.parquet
  python scripts/statcast_hitter_feature_pipeline.py --input path/to/statcast.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PCA / GMM (set True to run sklearn exploration block)
# ---------------------------------------------------------------------------
RUN_OPTIONAL_PCA_GMM = False

# ---------------------------------------------------------------------------
# Paths & naming (data set from 2020 to 2026 — not tied to a single season file)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "statcast_hitter_features_2020_2026"

# Description token sets (Statcast ``description`` strings)
_SWING = frozenset(
    {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "foul_bunt",
        "missed_bunt",
        "bunt_foul_tip",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
    }
)
_WHIFF = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
_CONTACT = frozenset(
    {
        "foul",
        "foul_tip",
        "foul_bunt",
        "bunt_foul_tip",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
    }
)
_BIP_DESC = frozenset({"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"})
_CALLED_STRIKE = frozenset({"called_strike"})
_BALL_DESC = frozenset({"ball", "blocked_ball", "pitchout"})
_FOUL = frozenset({"foul", "foul_tip", "foul_bunt", "bunt_foul_tip"})

_FASTBALL = frozenset({"FF", "SI", "FC", "FA"})
_BREAKING = frozenset({"SL", "ST", "CU", "KC", "SV"})
_OFFSPEED = frozenset({"CH", "FS", "FO", "SC", "EP"})

_AHEAD = frozenset({(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)})
_EVEN = frozenset({(0, 0), (1, 1), (2, 2), (3, 2)})
_BEHIND = frozenset({(0, 1), (0, 2), (1, 2)})

SAMPLE_COLS_FOR_CLUSTERING_EXCLUSION = frozenset(
    {
        "pitches_seen",
        "swings",
        "takes",
        "balls_in_play",
        "batted_ball_events",
        "pa_approx",
    }
)

# Raw pitch-count columns from ``_grouped_feature_frame`` (excluded from PCA/GMM with any suffix).
_RAW_COUNT_STEMS = frozenset(
    {
        "pitches_seen",
        "swings",
        "takes",
        "whiffs",
        "contacts",
        "balls_in_play",
        "batted_ball_events",
        "fouls",
        "called_strikes",
        "balls_called",
        "pitches_in_zone",
        "pitches_out_zone",
        "swings_in_zone",
        "swings_out_zone",
        "first_pitches",
        "swings_first_pitch",
        "two_strike_pitches",
        "swings_two_strike",
        "ahead_pitches",
        "swings_ahead",
        "even_pitches",
        "swings_even",
        "behind_pitches",
        "swings_behind",
        "contact_swings_in_zone",
        "contact_swings_out_zone",
        "whiffs_in_zone",
        "whiffs_out_zone",
        "takes_in_zone",
        "takes_out_zone",
        "called_strikes_in_zone",
    }
)

_WIDE_SUFFIXES = (
    "_fastball",
    "_breaking",
    "_offspeed",
    "_other",
    "_vs_RHP",
    "_vs_LHP",
    "_fastball_vs_RHP",
    "_fastball_vs_LHP",
    "_breaking_vs_RHP",
    "_breaking_vs_LHP",
    "_offspeed_vs_RHP",
    "_offspeed_vs_LHP",
    "_other_vs_RHP",
    "_other_vs_LHP",
)


def _warn(msg: str) -> None:
    warnings.warn(msg, UserWarning, stacklevel=2)
    print(f"[WARN] {msg}", flush=True)


def _norm_desc(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def map_pitch_family(pitch_type: pd.Series) -> pd.Series:
    """Map Statcast ``pitch_type`` codes to pitch_family labels."""
    pt = pitch_type.astype(str).str.strip().str.upper()
    out = pd.Series("other", index=pitch_type.index, dtype="object")
    out[pt.isin(_FASTBALL)] = "fastball"
    out[pt.isin(_BREAKING)] = "breaking"
    out[pt.isin(_OFFSPEED)] = "offspeed"
    out[pt.isin(["", "NAN", "NONE"]) | pt.isna()] = "other"
    return out


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dtypes and validate essential columns for the data set from 2020 to 2026."""
    need = ["batter", "description", "balls", "strikes"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(
            f"Essential columns missing for hitter pipeline (data set from 2020 to 2026): {miss}"
        )
    out = df.copy()
    out["batter"] = pd.to_numeric(out["batter"], errors="coerce")
    bad_b = out["batter"].isna().sum()
    if bad_b:
        _warn(f"Dropping {int(bad_b)} rows with non-numeric batter id.")
        out = out.loc[out["batter"].notna()].copy()
    out["balls"] = pd.to_numeric(out["balls"], errors="coerce").fillna(-1).astype(np.int64)
    out["strikes"] = pd.to_numeric(out["strikes"], errors="coerce").fillna(-1).astype(np.int64)
    out["_desc"] = _norm_desc(out["description"])
    return out


def add_helper_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add swing / contact / count / zone / pitch-family flags to pitch-level data
    from the data set from 2020 to 2026.
    """
    out = df.copy()
    d = out["_desc"]

    out["_swing"] = d.isin(_SWING)
    out["_whiff"] = d.isin(_WHIFF)
    out["_contact"] = d.isin(_CONTACT)
    out["_bip_desc"] = d.isin(_BIP_DESC)
    ls = pd.to_numeric(out["launch_speed"], errors="coerce") if "launch_speed" in out.columns else pd.Series(np.nan, index=out.index)
    la = pd.to_numeric(out["launch_angle"], errors="coerce") if "launch_angle" in out.columns else pd.Series(np.nan, index=out.index)
    out["_bbe"] = ls.notna() | la.notna()

    out["_called_strike"] = d.isin(_CALLED_STRIKE)
    out["_ball_desc"] = d.isin(_BALL_DESC)
    out["_foul"] = d.isin(_FOUL)
    out["_take"] = ~out["_swing"] & d.notna() & (d != "") & (d != "nan")

    b, s = out["balls"], out["strikes"]
    out["_first_pitch"] = (b == 0) & (s == 0)
    out["_two_strike"] = s == 2
    cs = list(zip(b.tolist(), s.tolist()))
    out["_ahead"] = pd.Series([t in _AHEAD for t in cs], index=out.index)
    out["_even"] = pd.Series([t in _EVEN for t in cs], index=out.index)
    out["_behind"] = pd.Series([t in _BEHIND for t in cs], index=out.index)

    if "pitch_type" in out.columns:
        out["pitch_family"] = map_pitch_family(out["pitch_type"])
    else:
        out["pitch_family"] = "other"
        _warn("Column pitch_type missing; pitch_family set to 'other' for all rows.")

    # Zone flags
    out["_in_zone"] = False
    out["_out_zone"] = False
    out["_zone_ok"] = False

    if "zone" in out.columns:
        z = pd.to_numeric(out["zone"], errors="coerce")
        out["_in_zone"] = z.isin(range(1, 10)).fillna(False).to_numpy(dtype=bool)
        out["_out_zone"] = z.isin([11, 12, 13, 14]).fillna(False).to_numpy(dtype=bool)
        out["_zone_ok"] = out["_in_zone"] | out["_out_zone"]
        out["_in_zone"] = out["_in_zone"].astype(bool)
        out["_out_zone"] = out["_out_zone"].astype(bool)
    elif all(c in out.columns for c in ("plate_x", "plate_z", "sz_top", "sz_bot")):
        px = pd.to_numeric(out["plate_x"], errors="coerce")
        pz = pd.to_numeric(out["plate_z"], errors="coerce")
        szt = pd.to_numeric(out["sz_top"], errors="coerce")
        szb = pd.to_numeric(out["sz_bot"], errors="coerce")
        out["_in_zone"] = (px >= -0.83) & (px <= 0.83) & (pz >= szb) & (pz <= szt)
        out["_in_zone"] = out["_in_zone"].fillna(False).astype(bool)
        out["_out_zone"] = ~out["_in_zone"] & px.notna() & pz.notna() & szt.notna() & szb.notna()
        out["_zone_ok"] = out["_in_zone"] | out["_out_zone"]
        _warn("Column zone missing; approximating in_zone / out_zone from plate_x, plate_z, sz_top, sz_bot.")
    else:
        _warn("Zone-like features skipped: need zone or (plate_x, plate_z, sz_top, sz_bot).")

    # High / low pitch (for swing profile)
    out["_high_pitch"] = False
    out["_low_pitch"] = False
    if "plate_z" in out.columns:
        pz = pd.to_numeric(out["plate_z"], errors="coerce")
        if all(c in out.columns for c in ("sz_top", "sz_bot")):
            szt = pd.to_numeric(out["sz_top"], errors="coerce")
            szb = pd.to_numeric(out["sz_bot"], errors="coerce")
            mid = (szt + szb) / 2.0
            span = (szt - szb).clip(lower=0.1) / 2.0
            out["_high_pitch"] = (pz > mid + 0.25 * span).fillna(False)
            out["_low_pitch"] = (pz < mid - 0.25 * span).fillna(False)
        else:
            out["_high_pitch"] = (pz > 2.5).fillna(False)
            out["_low_pitch"] = (pz < 1.5).fillna(False)

    # Inside / outside (hitter perspective)
    out["_inside"] = False
    out["_outside"] = False
    if "stand" in out.columns and "plate_x" in out.columns:
        px = pd.to_numeric(out["plate_x"], errors="coerce")
        st = out["stand"].astype(str).str.upper().str.strip()
        rhh = st == "R"
        lhh = st == "L"
        out["_inside"] = ((rhh & (px < 0)) | (lhh & (px > 0))).fillna(False)
        out["_outside"] = ((rhh & (px > 0)) | (lhh & (px < 0))).fillna(False)
    else:
        _warn("stand and/or plate_x missing; inside/outside location rates skipped.")

    return out


def _safe_div(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def _grouped_feature_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Core numeric aggregates for one grouping (e.g. ['batter'] or ['batter','pitch_family'])."""
    g = df.groupby(group_cols, sort=False)
    n = g.size().rename("pitches_seen")

    def _sum_mask(mask: pd.Series, name: str) -> pd.Series:
        return df.assign(__agg_mask=mask.astype(np.int32)).groupby(group_cols, sort=False)["__agg_mask"].sum().rename(name)

    swings = g["_swing"].sum().rename("swings")
    takes = g["_take"].sum().rename("takes")
    whiffs = g["_whiff"].sum().rename("whiffs")
    contacts = g["_contact"].sum().rename("contacts")
    bip = g["_bip_desc"].sum().rename("balls_in_play")
    bbe = g["_bbe"].sum().rename("batted_ball_events")
    fouls = g["_foul"].sum().rename("fouls")
    called = g["_called_strike"].sum().rename("called_strikes")
    balls = g["_ball_desc"].sum().rename("balls_called")

    in_z = g["_in_zone"].sum().rename("pitches_in_zone")
    out_z = g["_out_zone"].sum().rename("pitches_out_zone")
    swing_in = _sum_mask(df["_swing"] & df["_in_zone"], "swings_in_zone")
    swing_out = _sum_mask(df["_swing"] & df["_out_zone"], "swings_out_zone")

    fp = g["_first_pitch"].sum().rename("first_pitches")
    sw_fp = _sum_mask(df["_swing"] & df["_first_pitch"], "swings_first_pitch")

    ts = g["_two_strike"].sum().rename("two_strike_pitches")
    sw_ts = _sum_mask(df["_swing"] & df["_two_strike"], "swings_two_strike")

    ah = g["_ahead"].sum().rename("ahead_pitches")
    sw_ah = _sum_mask(df["_swing"] & df["_ahead"], "swings_ahead")

    ev = g["_even"].sum().rename("even_pitches")
    sw_ev = _sum_mask(df["_swing"] & df["_even"], "swings_even")

    bh = g["_behind"].sum().rename("behind_pitches")
    sw_bh = _sum_mask(df["_swing"] & df["_behind"], "swings_behind")

    cont_in = _sum_mask(df["_contact"] & df["_swing"] & df["_in_zone"], "contact_swings_in_zone")
    cont_out = _sum_mask(df["_contact"] & df["_swing"] & df["_out_zone"], "contact_swings_out_zone")

    wh_in = _sum_mask(df["_whiff"] & df["_swing"] & df["_in_zone"], "whiffs_in_zone")
    wh_out = _sum_mask(df["_whiff"] & df["_swing"] & df["_out_zone"], "whiffs_out_zone")

    take_in = _sum_mask(df["_take"] & df["_in_zone"], "takes_in_zone")
    take_out = _sum_mask(df["_take"] & df["_out_zone"], "takes_out_zone")

    cs_in = _sum_mask(df["_called_strike"] & df["_in_zone"], "called_strikes_in_zone")

    base = pd.concat(
        [
            n,
            swings,
            takes,
            whiffs,
            contacts,
            bip,
            bbe,
            fouls,
            called,
            balls,
            in_z,
            out_z,
            swing_in,
            swing_out,
            fp,
            sw_fp,
            ts,
            sw_ts,
            ah,
            sw_ah,
            ev,
            sw_ev,
            bh,
            sw_bh,
            cont_in,
            cont_out,
            wh_in,
            wh_out,
            take_in,
            take_out,
            cs_in,
        ],
        axis=1,
    )

    ps = base["pitches_seen"].to_numpy(dtype=float)
    sw = base["swings"].to_numpy(dtype=float)

    base["swing_rate"] = _safe_div(sw, ps)
    base["take_rate"] = _safe_div(base["takes"].to_numpy(dtype=float), ps)
    base["zone_swing_rate"] = _safe_div(base["swings_in_zone"].to_numpy(dtype=float), base["pitches_in_zone"].to_numpy(dtype=float))
    base["chase_rate"] = _safe_div(base["swings_out_zone"].to_numpy(dtype=float), base["pitches_out_zone"].to_numpy(dtype=float))
    base["first_pitch_swing_rate"] = _safe_div(base["swings_first_pitch"].to_numpy(dtype=float), base["first_pitches"].to_numpy(dtype=float))
    base["two_strike_swing_rate"] = _safe_div(base["swings_two_strike"].to_numpy(dtype=float), base["two_strike_pitches"].to_numpy(dtype=float))
    base["ahead_count_swing_rate"] = _safe_div(base["swings_ahead"].to_numpy(dtype=float), base["ahead_pitches"].to_numpy(dtype=float))
    base["even_count_swing_rate"] = _safe_div(base["swings_even"].to_numpy(dtype=float), base["even_pitches"].to_numpy(dtype=float))
    base["behind_count_swing_rate"] = _safe_div(base["swings_behind"].to_numpy(dtype=float), base["behind_pitches"].to_numpy(dtype=float))

    base["contact_rate"] = _safe_div(base["contacts"].to_numpy(dtype=float), sw)
    base["whiff_rate"] = _safe_div(base["whiffs"].to_numpy(dtype=float), sw)
    base["zone_contact_rate"] = _safe_div(base["contact_swings_in_zone"].to_numpy(dtype=float), base["swings_in_zone"].to_numpy(dtype=float))
    base["chase_contact_rate"] = _safe_div(base["contact_swings_out_zone"].to_numpy(dtype=float), base["swings_out_zone"].to_numpy(dtype=float))
    base["zone_whiff_rate"] = _safe_div(base["whiffs_in_zone"].to_numpy(dtype=float), base["swings_in_zone"].to_numpy(dtype=float))
    base["chase_whiff_rate"] = _safe_div(base["whiffs_out_zone"].to_numpy(dtype=float), base["swings_out_zone"].to_numpy(dtype=float))
    base["foul_rate_per_swing"] = _safe_div(base["fouls"].to_numpy(dtype=float), sw)
    base["bip_rate_per_swing"] = _safe_div(base["balls_in_play"].to_numpy(dtype=float), sw)
    csw_num = base["called_strikes"].to_numpy(dtype=float) + base["whiffs"].to_numpy(dtype=float)
    base["csw_rate"] = _safe_div(csw_num, ps)

    base["called_strike_rate"] = _safe_div(base["called_strikes"].to_numpy(dtype=float), ps)
    base["called_strike_zone_rate"] = _safe_div(base["called_strikes_in_zone"].to_numpy(dtype=float), base["pitches_in_zone"].to_numpy(dtype=float))
    base["o_take_rate"] = _safe_div(base["takes_out_zone"].to_numpy(dtype=float), base["pitches_out_zone"].to_numpy(dtype=float))
    base["z_take_rate"] = _safe_div(base["takes_in_zone"].to_numpy(dtype=float), base["pitches_in_zone"].to_numpy(dtype=float))

    return base.reset_index()


def _strip_wide_suffix(col: str) -> str:
    stem = col
    for suf in sorted(_WIDE_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def _bip_quality_by_batter(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """EV / LA / xStats on batted-ball-event rows (launch_speed or launch_angle present)."""
    sub = df.loc[df["_bbe"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=group_cols)

    gcols = group_cols
    g = sub.groupby(gcols, sort=False)
    merged: pd.DataFrame | None = None

    if "launch_speed" in sub.columns:
        sub["_ls"] = pd.to_numeric(sub["launch_speed"], errors="coerce")
        agg_ls = g["_ls"].agg(
            avg_ev="mean",
            median_ev="median",
            ev_std="std",
            max_ev="max",
            ev75=lambda s: s.quantile(0.75),
            ev90=lambda s: s.quantile(0.9),
            ev95=lambda s: s.quantile(0.95),
        )
        hh = sub.assign(__hh=((sub["_ls"] >= 95).fillna(False).astype(np.int32))).groupby(gcols, sort=False)["__hh"].sum()
        sc = sub.assign(__sc=((sub["_ls"] < 80).fillna(False).astype(np.int32))).groupby(gcols, sort=False)["__sc"].sum()
        nbb = sub.groupby(gcols, sort=False)["_ls"].count()
        agg_ls = agg_ls.copy()
        agg_ls["hard_hit_rate"] = _safe_div(hh.to_numpy(dtype=float), nbb.to_numpy(dtype=float))
        agg_ls["soft_contact_rate"] = _safe_div(sc.to_numpy(dtype=float), nbb.to_numpy(dtype=float))
        merged = agg_ls.reset_index()
    else:
        _warn("launch_speed missing; EV / hard-hit contact-quality block skipped for this split.")

    if "launch_angle" in sub.columns:
        sub["_la"] = pd.to_numeric(sub["launch_angle"], errors="coerce")
        agg_la = g["_la"].agg(
            avg_la="mean",
            median_la="median",
            la_std="std",
            la10=lambda s: s.quantile(0.1),
            la25=lambda s: s.quantile(0.25),
            la75=lambda s: s.quantile(0.75),
            la90=lambda s: s.quantile(0.9),
        )
        nla = sub.groupby(gcols)["_la"].count().to_numpy(dtype=float)
        sweet = (
            sub.assign(__s=(((sub["_la"] >= 8) & (sub["_la"] <= 32)).fillna(False).astype(np.int32)))
            .groupby(gcols, sort=False)["__s"]
            .sum()
            .to_numpy(dtype=float)
        )
        gb = (
            sub.assign(__g=((sub["_la"] < 10).fillna(False).astype(np.int32)))
            .groupby(gcols, sort=False)["__g"]
            .sum()
            .to_numpy(dtype=float)
        )
        ld = (
            sub.assign(__l=(((sub["_la"] >= 10) & (sub["_la"] <= 25)).fillna(False).astype(np.int32)))
            .groupby(gcols, sort=False)["__l"]
            .sum()
            .to_numpy(dtype=float)
        )
        fb = (
            sub.assign(__f=(((sub["_la"] > 25) & (sub["_la"] <= 50)).fillna(False).astype(np.int32)))
            .groupby(gcols, sort=False)["__f"]
            .sum()
            .to_numpy(dtype=float)
        )
        pu = (
            sub.assign(__p=((sub["_la"] > 50).fillna(False).astype(np.int32)))
            .groupby(gcols, sort=False)["__p"]
            .sum()
            .to_numpy(dtype=float)
        )
        agg_la = agg_la.copy()
        agg_la["sweet_spot_rate"] = _safe_div(sweet, nla)
        agg_la["ground_ball_rate"] = _safe_div(gb, nla)
        agg_la["line_drive_rate"] = _safe_div(ld, nla)
        agg_la["fly_ball_rate"] = _safe_div(fb, nla)
        agg_la["popup_rate"] = _safe_div(pu, nla)
        agg_la = agg_la.reset_index()
        merged = agg_la if merged is None else merged.merge(agg_la, on=gcols, how="outer")
    else:
        _warn("launch_angle missing; LA / batted-ball-shape block skipped for this split.")

    if "launch_speed_angle" in sub.columns:
        sub["_lsa"] = pd.to_numeric(sub["launch_speed_angle"], errors="coerce")
        idx = sub.groupby(gcols, sort=False).size().index
        totals = sub.groupby(gcols, sort=False)["_lsa"].count().replace(0, np.nan)
        lsa_rates: dict[str, np.ndarray] = {}
        for val, name in [
            (1, "weak_rate"),
            (2, "topped_rate"),
            (3, "under_rate"),
            (4, "flare_burner_rate"),
            (5, "solid_contact_rate"),
            (6, "barrel_rate"),
        ]:
            cnt = (
                sub.assign(__c=((sub["_lsa"] == val).fillna(False).astype(np.int32)))
                .groupby(gcols, sort=False)["__c"]
                .sum()
                .reindex(idx, fill_value=0)
            )
            lsa_rates[name] = _safe_div(cnt.to_numpy(dtype=float), totals.reindex(idx).to_numpy(dtype=float))
        lsa_df = pd.DataFrame(lsa_rates, index=idx).reset_index()
        merged = lsa_df if merged is None else merged.merge(lsa_df, on=gcols, how="outer")

    if "estimated_ba_using_speedangle" in sub.columns:
        sub["_xba"] = pd.to_numeric(sub["estimated_ba_using_speedangle"], errors="coerce")
    if "estimated_woba_using_speedangle" in sub.columns:
        sub["_xw"] = pd.to_numeric(sub["estimated_woba_using_speedangle"], errors="coerce")

    if "estimated_ba_using_speedangle" in sub.columns or "estimated_woba_using_speedangle" in sub.columns:
        g2 = sub.groupby(gcols, sort=False)
        xparts2: list[pd.DataFrame] = []
        if "estimated_ba_using_speedangle" in sub.columns:
            xparts2.append(
                g2["_xba"].agg(mean_xba_contact="mean", xba_90=lambda s: s.quantile(0.9)).reset_index()
            )
        if "estimated_woba_using_speedangle" in sub.columns:
            xparts2.append(
                g2["_xw"].agg(
                    mean_xwoba_contact="mean",
                    xwoba_90=lambda s: s.quantile(0.9),
                    std_xwoba_contact="std",
                ).reset_index()
            )
        xdf = xparts2[0]
        for xp in xparts2[1:]:
            xdf = xdf.merge(xp, on=gcols, how="outer")
        merged = xdf if merged is None else merged.merge(xdf, on=gcols, how="outer")

    if merged is None:
        return pd.DataFrame(columns=group_cols)
    return merged


def _location_profile_by_batter(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Plate location means and directional swing/whiff rates."""
    need = {"plate_x", "plate_z"}
    if not need.issubset(df.columns):
        _warn("plate_x / plate_z missing; location-based swing profile skipped.")
        return pd.DataFrame(columns=group_cols)

    px = pd.to_numeric(df["plate_x"], errors="coerce")
    pz = pd.to_numeric(df["plate_z"], errors="coerce")

    g = df.assign(_px=px, _pz=pz)

    def _loc_block(x: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "mean_plate_x_swung": x.loc[x["_swing"], "_px"].mean(),
                "mean_plate_z_swung": x.loc[x["_swing"], "_pz"].mean(),
                "std_plate_x_swung": x.loc[x["_swing"], "_px"].std(),
                "std_plate_z_swung": x.loc[x["_swing"], "_pz"].std(),
                "mean_plate_x_whiff": x.loc[x["_whiff"], "_px"].mean(),
                "mean_plate_z_whiff": x.loc[x["_whiff"], "_pz"].mean(),
                "mean_plate_x_contact": x.loc[x["_contact"], "_px"].mean(),
                "mean_plate_z_contact": x.loc[x["_contact"], "_pz"].mean(),
            }
        )

    grp = g.groupby(group_cols, sort=False)
    try:
        out = grp.apply(_loc_block, include_groups=False)
    except TypeError:
        out = grp.apply(_loc_block)
    out = out.reset_index()

    def _sum_mask2(mask: pd.Series) -> pd.Series:
        return df.assign(__m=mask.astype(np.int32)).groupby(group_cols, sort=False)["__m"].sum()

    hi_sw = _sum_mask2(df["_swing"] & df["_high_pitch"])
    lo_sw = _sum_mask2(df["_swing"] & df["_low_pitch"])
    hi_wf = _sum_mask2(df["_whiff"] & df["_high_pitch"])
    lo_wf = _sum_mask2(df["_whiff"] & df["_low_pitch"])
    in_sw = _sum_mask2(df["_swing"] & df["_inside"])
    out_sw = _sum_mask2(df["_swing"] & df["_outside"])
    in_wf = _sum_mask2(df["_whiff"] & df["_inside"])
    out_wf = _sum_mask2(df["_whiff"] & df["_outside"])
    sw_n = df.groupby(group_cols, sort=False)["_swing"].sum()
    wf_n = df.groupby(group_cols, sort=False)["_whiff"].sum()

    loc_rates = pd.DataFrame(
        {
            "high_swing_rate": _safe_div(hi_sw.to_numpy(dtype=float), sw_n.to_numpy(dtype=float)),
            "low_swing_rate": _safe_div(lo_sw.to_numpy(dtype=float), sw_n.to_numpy(dtype=float)),
            "inside_swing_rate": _safe_div(in_sw.to_numpy(dtype=float), sw_n.to_numpy(dtype=float)),
            "outside_swing_rate": _safe_div(out_sw.to_numpy(dtype=float), sw_n.to_numpy(dtype=float)),
            "high_whiff_rate": _safe_div(hi_wf.to_numpy(dtype=float), wf_n.to_numpy(dtype=float)),
            "low_whiff_rate": _safe_div(lo_wf.to_numpy(dtype=float), wf_n.to_numpy(dtype=float)),
            "inside_whiff_rate": _safe_div(in_wf.to_numpy(dtype=float), wf_n.to_numpy(dtype=float)),
            "outside_whiff_rate": _safe_div(out_wf.to_numpy(dtype=float), wf_n.to_numpy(dtype=float)),
        },
        index=sw_n.index,
    ).reset_index()

    return out.merge(loc_rates, on=group_cols, how="outer")


def build_split_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Rates + BIP quality + location for one stratification."""
    base = _grouped_feature_frame(df, group_cols)
    bip = _bip_quality_by_batter(df, group_cols)
    loc = _location_profile_by_batter(df, group_cols)

    mk = group_cols
    out = base
    if not bip.empty and len(bip.columns) > len(mk):
        out = out.merge(bip, on=mk, how="left")
    if not loc.empty and len(loc.columns) > len(mk):
        out = out.merge(loc, on=mk, how="left")
    return out


def long_to_wide(long_df: pd.DataFrame, batter_col: str, facet_col: str, suffix_map: dict[str, str]) -> pd.DataFrame:
    """Pivot facet_col levels to wide columns with suffixes (e.g. fastball -> _fastball)."""
    if long_df.empty:
        return pd.DataFrame(columns=[batter_col])
    feat_cols = [c for c in long_df.columns if c not in (batter_col, facet_col)]
    parts = [long_df[[batter_col]].drop_duplicates()]
    for level, suf in suffix_map.items():
        sub = long_df.loc[long_df[facet_col] == level, [batter_col] + feat_cols]
        sub = sub.rename(columns={c: f"{c}{suf}" for c in feat_cols})
        parts.append(sub)
    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on=batter_col, how="outer")
    return wide


def add_pa_approx(df: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Append pa_approx per batter if ``events`` exists."""
    out = overall.copy()
    if "events" not in df.columns:
        _warn("events column missing; pa_approx not computed.")
        return out
    ev = df["events"]
    mask = ev.notna() & (ev.astype(str).str.strip() != "") & (ev.astype(str).str.lower() != "nan")
    pa = mask.groupby(df["batter"], sort=False).sum().rename("pa_approx").reset_index()
    return out.merge(pa, on="batter", how="left")


def build_clustering_columns(modeling: pd.DataFrame) -> list[str]:
    """Columns safe for PCA/GMM: exclude ids, names, season, sample sizes."""
    id_like = {
        "batter",
        "player_name",
        "game_year",
        "game_year_min",
        "game_year_max",
        "pitch_family",
        "p_throws",
    }
    cols: list[str] = []
    for c in modeling.columns:
        if c in id_like:
            continue
        if c.startswith("n_"):
            continue
        stem = _strip_wide_suffix(c)
        if stem in _RAW_COUNT_STEMS or stem in SAMPLE_COLS_FOR_CLUSTERING_EXCLUSION:
            continue
        if pd.api.types.is_numeric_dtype(modeling[c]):
            cols.append(c)
    return sorted(cols)


def run_optional_pca_gmm(modeling: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    use_cols = [c for c in feature_cols if c in modeling.columns]
    if len(use_cols) < 3:
        _warn("Fewer than 3 clustering columns present; skipping PCA/GMM.")
        return
    X = modeling[use_cols].to_numpy(dtype=float)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(15, X.shape[1], X.shape[0]), random_state=42)),
        ]
    )
    Z = pipe.fit_transform(X)
    pca_scores = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])])
    pca_scores.insert(0, "batter", modeling["batter"].values)
    pca_scores.to_csv(out_dir / "pca_scores.csv", index=False)

    rows = []
    assigns = pd.DataFrame({"batter": modeling["batter"].values})
    for k in range(3, 11):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, max_iter=200)
        gmm.fit(Z)
        rows.append({"K": k, "bic": gmm.bic(Z), "aic": gmm.aic(Z)})
        assigns[f"gmm_k{k}"] = gmm.predict(Z)
    pd.DataFrame(rows).to_csv(out_dir / "gmm_model_selection.csv", index=False)
    assigns.to_csv(out_dir / "gmm_cluster_assignments.csv", index=False)
    print("[PCA/GMM] Saved pca_scores.csv, gmm_model_selection.csv, gmm_cluster_assignments.csv", flush=True)


def run_pipeline(
    *,
    df: pd.DataFrame | None = None,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    single_output: Path | None = None,
) -> Path:
    """
    End-to-end feature build from the data set from 2020 to 2026.

    Provide either ``df`` or ``input_path`` (CSV or Parquet).

    If ``single_output`` is set, writes **only** that file (merged wide modeling
    table: one row per batter). Suffix ``.parquet`` or ``.csv`` selects format.
    Intermediate CSVs are not written.
    """
    single_output = Path(single_output).resolve() if single_output is not None else None
    write_intermediates = single_output is None
    out_dir = (
        (output_dir or DEFAULT_OUTPUT_DIR).resolve()
        if write_intermediates
        else single_output.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_intermediates:
        print(f"[pipeline] Output directory (data set from 2020 to 2026 features): {out_dir}", flush=True)
    else:
        print(f"[pipeline] Single-output mode -> {single_output}", flush=True)

    if df is None:
        if input_path is None:
            raise ValueError("Provide df= or input_path=")
        input_path = Path(input_path)
        print(f"[pipeline] Loading {input_path}", flush=True)
        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix.lower() in (".csv", ".txt"):
            df = pd.read_csv(input_path, low_memory=False)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")

    print(f"[pipeline] Input rows: {len(df):,}", flush=True)
    df = validate_and_clean(df)
    df = add_helper_flags(df)

    # --- Overall ---
    print("[pipeline] Building hitter_overall_features …", flush=True)
    overall = build_split_table(df, ["batter"])
    overall = add_pa_approx(df, overall)

    if "player_name" in df.columns:
        names = df.sort_values("game_date") if "game_date" in df.columns else df
        pname = names.groupby("batter", sort=False)["player_name"].last().reset_index()
        overall = overall.merge(pname, on="batter", how="left")

    if write_intermediates:
        overall.to_csv(out_dir / "hitter_overall_features.csv", index=False)

    # --- pitch_family splits ---
    print("[pipeline] Building pitch_family tables …", flush=True)
    pf_long = build_split_table(df, ["batter", "pitch_family"])
    if write_intermediates:
        pf_long.to_csv(out_dir / "hitter_pitch_family_features_long.csv", index=False)

    fam_map = {"fastball": "_fastball", "breaking": "_breaking", "offspeed": "_offspeed", "other": "_other"}
    pf_wide = long_to_wide(pf_long, "batter", "pitch_family", fam_map)
    if write_intermediates:
        pf_wide.to_csv(out_dir / "hitter_pitch_family_features_wide.csv", index=False)

    # --- pitcher hand ---
    if "p_throws" in df.columns:
        hand = df.loc[df["p_throws"].isin(["R", "L"])].copy()
        ph_long = build_split_table(hand, ["batter", "p_throws"])
        if write_intermediates:
            ph_long.to_csv(out_dir / "hitter_pitcher_hand_features_long.csv", index=False)
        ph_wide = long_to_wide(ph_long, "batter", "p_throws", {"R": "_vs_RHP", "L": "_vs_LHP"})
        if write_intermediates:
            ph_wide.to_csv(out_dir / "hitter_pitcher_hand_features_wide.csv", index=False)
    else:
        _warn("p_throws missing; pitcher-hand split tables skipped.")
        ph_long = pd.DataFrame()
        ph_wide = pd.DataFrame()

    # --- family x hand ---
    if "p_throws" in df.columns:
        fh = df.loc[df["p_throws"].isin(["R", "L"])].copy()
        fh["_facet"] = fh["pitch_family"].astype(str) + "_vs_" + fh["p_throws"].map({"R": "RHP", "L": "LHP"})
        fhand_long = build_split_table(fh, ["batter", "_facet"])
        fhand_long = fhand_long.rename(columns={"_facet": "pitch_family_hand"})
        if write_intermediates:
            fhand_long.to_csv(out_dir / "hitter_pitch_family_hand_features_long.csv", index=False)
        facet_suffix = {
            "fastball_vs_RHP": "_fastball_vs_RHP",
            "fastball_vs_LHP": "_fastball_vs_LHP",
            "breaking_vs_RHP": "_breaking_vs_RHP",
            "breaking_vs_LHP": "_breaking_vs_LHP",
            "offspeed_vs_RHP": "_offspeed_vs_RHP",
            "offspeed_vs_LHP": "_offspeed_vs_LHP",
            "other_vs_RHP": "_other_vs_RHP",
            "other_vs_LHP": "_other_vs_LHP",
        }
        fhand_wide = long_to_wide(fhand_long, "batter", "pitch_family_hand", facet_suffix)
        if write_intermediates:
            fhand_wide.to_csv(out_dir / "hitter_pitch_family_hand_features_wide.csv", index=False)
    else:
        fhand_long = pd.DataFrame()
        fhand_wide = pd.DataFrame()

    # --- Modeling merge ---
    print("[pipeline] Merging modeling table …", flush=True)
    modeling = overall.merge(pf_wide, on="batter", how="left", suffixes=("", "_dup"))
    modeling = modeling.loc[:, [c for c in modeling.columns if not c.endswith("_dup")]]
    if not ph_wide.empty:
        modeling = modeling.merge(ph_wide, on="batter", how="left")
    if not fhand_wide.empty:
        modeling = modeling.merge(fhand_wide, on="batter", how="left")

    if "game_year" in df.columns:
        gy = pd.to_numeric(df["game_year"], errors="coerce")
        modeling = modeling.merge(
            df.assign(_gy=gy).groupby("batter", sort=False)["_gy"].agg(game_year_min="min", game_year_max="max").reset_index(),
            on="batter",
            how="left",
        )

    if single_output is not None:
        suf = single_output.suffix.lower()
        if suf == ".parquet":
            modeling.to_parquet(single_output, index=False)
        elif suf in (".csv", ".txt"):
            modeling.to_csv(single_output, index=False)
        else:
            raise ValueError("--single-output must end with .parquet, .csv, or .txt")
        print(f"[pipeline] Wrote feature dataset ({len(modeling):,} rows, {len(modeling.columns)} cols): {single_output}", flush=True)
    else:
        modeling.to_csv(out_dir / "hitter_modeling_features.csv", index=False)

    filt = modeling.loc[
        (modeling["pitches_seen"] >= 600)
        & (modeling["swings"] >= 250)
        & (modeling["batted_ball_events"] >= 75)
    ].copy()
    if write_intermediates:
        filt.to_csv(out_dir / "hitter_modeling_features_filtered.csv", index=False)

    cluster_cols = build_clustering_columns(modeling)
    globals()["CLUSTERING_FEATURE_COLUMNS"] = cluster_cols
    if write_intermediates:
        (out_dir / "clustering_feature_columns.json").write_text(
            json.dumps(cluster_cols, indent=2),
            encoding="utf-8",
        )

    if RUN_OPTIONAL_PCA_GMM and write_intermediates:
        print("[pipeline] RUN_OPTIONAL_PCA_GMM is True; running PCA/GMM …", flush=True)
        run_optional_pca_gmm(filt, cluster_cols, out_dir)
    elif RUN_OPTIONAL_PCA_GMM and not write_intermediates:
        _warn("RUN_OPTIONAL_PCA_GMM is True but single-output mode skips PCA/GMM file writes.")
    else:
        print("[pipeline] RUN_OPTIONAL_PCA_GMM is False; skipping PCA/GMM.", flush=True)

    _print_sanity_summaries(overall, pf_long, ph_long, fhand_long, modeling, filt, cluster_cols)
    return single_output if single_output is not None else out_dir


def _print_sanity_summaries(
    overall: pd.DataFrame,
    pf_long: pd.DataFrame,
    ph_long: pd.DataFrame,
    fhand_long: pd.DataFrame,
    modeling: pd.DataFrame,
    filt: pd.DataFrame,
    cluster_cols: list[str],
) -> None:
    print("\n=== Sanity summary (data set from 2020 to 2026) ===", flush=True)
    print(f"Unique hitters (batter): {overall['batter'].nunique():,}", flush=True)
    print(f"hitter_overall_features.csv rows: {len(overall):,}", flush=True)
    print(f"hitter_pitch_family_features_long.csv rows: {len(pf_long):,}", flush=True)
    if not ph_long.empty:
        print(f"hitter_pitcher_hand_features_long.csv rows: {len(ph_long):,}", flush=True)
    if not fhand_long.empty:
        print(f"hitter_pitch_family_hand_features_long.csv rows: {len(fhand_long):,}", flush=True)
    print(f"hitter_modeling_features.csv rows: {len(modeling):,}", flush=True)
    print(f"hitter_modeling_features_filtered.csv rows: {len(filt):,}", flush=True)

    cc = [c for c in cluster_cols if c in modeling.columns]
    miss = modeling[cc].isna().mean().sort_values(ascending=False)
    print("\nTop 20 clustering features by missing share:", flush=True)
    print(miss.head(20).to_string(), flush=True)

    cov = modeling[cc].notna().mean().sort_values(ascending=False)
    print("\nTop 20 clustering features by non-null coverage:", flush=True)
    print(cov.head(20).to_string(), flush=True)


# Exported for download hook
CLUSTERING_FEATURE_COLUMNS: list[str] = []


def main() -> None:
    p = argparse.ArgumentParser(description="Statcast hitter feature pipeline (2020–2026 data).")
    p.add_argument("--input", type=Path, required=True, help="Parquet or CSV pitch-level Statcast file.")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: statcast_hitter_features_2020_2026 under project root.")
    p.add_argument(
        "--single-output",
        type=Path,
        default=None,
        help="Write only this merged modeling table (.parquet or .csv); skip other CSV/JSON outputs.",
    )
    args = p.parse_args()
    run_pipeline(input_path=args.input, output_dir=args.output_dir, single_output=args.single_output)


if __name__ == "__main__":
    main()
