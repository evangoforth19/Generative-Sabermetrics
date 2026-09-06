#!/usr/bin/env python3
"""
Fit separate Gaussian Mixture Models on the seven pitch-context features,
**within each coarse pitch-type group** (not by handedness).

Same 7D features as ``pitch_context_gmm.py`` / sbi_forward_sim numeric_context_features.

Pitch groups (Statcast ``pitch_type`` codes, uppercase):
  4F <- FF, FA | 2F <- SI, FT | CF <- FC | S <- SL, ST | C <- CU, KC, CS | CH <- CH, FS

Save fixed-K models only (omit full AIC/BIC grid):

  python scripts/pitch_context_gmm_pitch_groups.py --skip-aic-bic-grid \\
    --save-k-map '{"4F":10,"2F":9,"CF":8,"S":9,"C":9,"CH":12}' \\
    --n-init 10 --out-dir outputs/pitch_context_gmm_pitch_groups/fitted_k

Produces ``pitch_group_<GROUP>_k<K>.joblib`` (scaled 7-D features; invert with scaler).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

GROUP_MAP: dict[str, list[str]] = {
    "4F": ["FF", "FA"],
    "2F": ["SI", "FT"],
    "CF": ["FC"],
    "S": ["SL", "ST"],
    "C": ["CU", "KC", "CS"],
    "CH": ["CH", "FS"],
}

GROUP_ORDER = ("4F", "2F", "CF", "S", "C", "CH")

Z_COUNT_MAP = {
    "0-0": 0.3144746543,
    "0-1": 0.2704261285,
    "0-2": 0.2004055518,
    "1-0": 0.3612777549,
    "1-1": 0.3050139272,
    "1-2": 0.2260470281,
    "2-0": 0.4309212696,
    "2-1": 0.3613989554,
    "2-2": 0.2734161604,
    "3-0": 0.5416601816,
    "3-1": 0.4803511791,
    "3-2": 0.3824951644,
}

NUMERIC_CONTEXT_FEATURES = [
    "z_count",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "spin_axis_sin",
    "spin_axis_cos",
    "release_speed",
]


def _pitch_type_to_group_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for g, pts in GROUP_MAP.items():
        for pt in pts:
            out[pt.upper()] = g
    return out


PT_TO_GROUP = _pitch_type_to_group_map()


def _parse_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(date.fromisoformat(s.strip()))


def add_z_count(df: pd.DataFrame) -> pd.DataFrame:
    if "z_count" in df.columns and df["z_count"].notna().all():
        return df
    if "balls" not in df.columns or "strikes" not in df.columns:
        raise ValueError("Need balls/strikes to build z_count")
    out = df.copy()
    b = pd.to_numeric(out["balls"], errors="coerce").round().astype("Int64")
    s = pd.to_numeric(out["strikes"], errors="coerce").round().astype("Int64")
    cs = b.astype(str) + "-" + s.astype(str)
    out["z_count"] = cs.map(Z_COUNT_MAP)
    if out["z_count"].isna().any():
        bad = sorted(cs[out["z_count"].isna()].dropna().unique().tolist())[:20]
        raise ValueError(f"Unmapped count states for z_count: {bad[:10]}…")
    return out


def add_spin_trig(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "spin_axis" not in out.columns:
        raise ValueError("Statcast column spin_axis is required")
    rad = np.deg2rad(pd.to_numeric(out["spin_axis"], errors="coerce"))
    out["spin_axis_sin"] = np.sin(rad)
    out["spin_axis_cos"] = np.cos(rad)
    return out


def _parquet_columns() -> list[str]:
    return [
        "game_date",
        "p_throws",
        "pitch_type",
        "plate_x",
        "plate_z",
        "release_spin_rate",
        "release_speed",
        "spin_axis",
        "balls",
        "strikes",
    ]


def load_statcast_frames(paths: list[Path]) -> pd.DataFrame:
    frames = []
    cols = _parquet_columns()
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p, columns=cols))
    return pd.concat(frames, ignore_index=True)


def prepare_matrix_with_groups(df: pd.DataFrame):
    need_cols = ["game_date", "pitch_type", "plate_x", "plate_z", "release_spin_rate", "release_speed"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Input missing columns: {miss}")

    d = df.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d[d["game_date"].notna()].copy()

    d["p_throws"] = d["p_throws"].astype(str).str.strip().str.upper()
    d = d[d["p_throws"].isin(["L", "R"])].copy()

    pt = d["pitch_type"].astype(str).str.strip().str.upper()
    d["pitch_group"] = pt.map(PT_TO_GROUP)
    d = d[d["pitch_group"].notna()].copy()

    d = add_z_count(d)
    d = add_spin_trig(d)

    for c in NUMERIC_CONTEXT_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    before = len(d)
    d = d.dropna(subset=NUMERIC_CONTEXT_FEATURES)
    dropped_na_inner = before - len(d)

    X = d[NUMERIC_CONTEXT_FEATURES].to_numpy(dtype=np.float64)
    groups = d["pitch_group"].to_numpy()
    dates = d["game_date"].to_numpy()

    pt_clean = d["pitch_type"].astype(str).str.strip().str.upper()
    pitch_type_counts = pt_clean.value_counts().to_dict()

    return X, groups, dates, dropped_na_inner, len(d), pitch_type_counts


def fit_gmm_grid(
    X: np.ndarray,
    ks: range,
    *,
    seed: int,
    reg_covar: float,
    covariance_type: str,
    n_init: int,
) -> list[dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rows = []
    for k in ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=seed,
            reg_covar=reg_covar,
            max_iter=300,
            n_init=n_init,
        )
        gmm.fit(Xs)
        rows.append(
            {
                "k": k,
                "aic": float(gmm.aic(Xs)),
                "bic": float(gmm.bic(Xs)),
                "log_likelihood": float(gmm.score(Xs) * Xs.shape[0]),
                "n_iter": int(gmm.n_iter_),
                "converged": bool(gmm.converged_),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="GMM on 7 context features per pitch-type group")
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--parquet-glob", type=str, default=None)
    ap.add_argument("--date-start", type=str, default="2020-01-01")
    ap.add_argument("--date-end", type=str, default="2099-12-31")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows-per-group", type=int, default=None)
    ap.add_argument("--reg-covar", type=float, default=1e-4)
    ap.add_argument(
        "--covariance-type",
        choices=("full", "diag", "tied", "spherical"),
        default="diag",
    )
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument(
        "--save-k-map",
        type=str,
        default=None,
        help='JSON dict of pitch_group -> K, e.g. {"4F":10,"2F":9,"CF":8,"S":9,"C":9,"CH":12}',
    )
    ap.add_argument(
        "--save-k-map-file",
        type=Path,
        default=None,
        help="Path to JSON file (same format as --save-k-map); overrides --save-k-map if both set",
    )
    ap.add_argument(
        "--skip-aic-bic-grid",
        action="store_true",
        help="Skip fitting all K from k-min..k-max; use with --save-k-map for fast artifact-only runs",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/pitch_context_gmm_pitch_groups"))
    args = ap.parse_args()

    if args.parquet is None and args.parquet_glob is None:
        root = Path(__file__).resolve().parents[1]
        default_p = root / "data/statcast_pybaseball/statcast_all.parquet"
        if default_p.is_file():
            args.parquet = default_p
        else:
            print("Pass --parquet or --parquet-glob", file=sys.stderr)
            sys.exit(1)

    if args.parquet is not None:
        paths = [args.parquet]
    else:
        paths = sorted(Path(p) for p in glob(args.parquet_glob))
        if not paths:
            print(f"No files matched {args.parquet_glob!r}", file=sys.stderr)
            sys.exit(1)

    df = load_statcast_frames(paths)
    t0 = _parse_date(args.date_start)
    t1 = _parse_date(args.date_end)
    df = df[
        (pd.to_datetime(df["game_date"], errors="coerce") >= t0)
        & (pd.to_datetime(df["game_date"], errors="coerce") <= t1)
    ].copy()

    rng = np.random.default_rng(args.seed)

    X, groups, dates, dropped_na, n_used, pitch_type_counts = prepare_matrix_with_groups(df)

    report: dict = {
        "parquet_paths": [str(p) for p in paths],
        "date_start": args.date_start,
        "date_end": args.date_end,
        "game_date_min_observed": str(pd.to_datetime(dates).min()),
        "game_date_max_observed": str(pd.to_datetime(dates).max()),
        "n_rows_mapped_to_pitch_groups": int(n_used),
        "rows_dropped_missing_features_after_group_map": int(dropped_na),
        "group_map": GROUP_MAP,
        "pitch_type_counts_in_clean_sample": pitch_type_counts,
        "features": NUMERIC_CONTEXT_FEATURES,
        "covariance_type": args.covariance_type,
        "groups": {},
    }

    ks = range(args.k_min, args.k_max + 1)
    flat: list[dict] = []

    group_data: dict[str, np.ndarray] = {}

    for gname in GROUP_ORDER:
        Xm = X[groups == gname]
        if args.max_rows_per_group is not None and Xm.shape[0] > args.max_rows_per_group:
            idx = rng.choice(Xm.shape[0], size=args.max_rows_per_group, replace=False)
            Xm = Xm[idx]
        if Xm.shape[0] < args.min_rows:
            report["groups"][gname] = {
                "error": f"Too few rows ({Xm.shape[0]} < min_rows={args.min_rows})",
                "n_rows": int(Xm.shape[0]),
            }
            continue
        group_data[gname] = Xm
        report["groups"][gname] = {"n_rows": int(Xm.shape[0])}

    if not args.skip_aic_bic_grid:
        for gname, Xm in group_data.items():
            rows = fit_gmm_grid(
                Xm,
                ks,
                seed=args.seed,
                reg_covar=args.reg_covar,
                covariance_type=args.covariance_type,
                n_init=args.n_init,
            )
            report["groups"][gname]["aic_bic_by_k"] = rows
            for row in rows:
                flat.append({"pitch_group": gname, **row})

    saved_manifest: dict[str, dict] = {}
    if args.save_k_map_file is not None:
        k_map_raw = json.loads(Path(args.save_k_map_file).read_text(encoding="utf-8"))
    elif args.save_k_map is not None:
        k_map_raw = json.loads(args.save_k_map)
    else:
        k_map_raw = None

    if k_map_raw is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        k_map_parsed = {str(k): int(v) for k, v in k_map_raw.items()}
        unknown = set(k_map_parsed) - set(GROUP_ORDER)
        if unknown:
            print(f"Warning: ignoring unknown pitch_group keys {sorted(unknown)}", file=sys.stderr)
        n_init_fit = max(int(args.n_init), 6)
        for gname in GROUP_ORDER:
            if gname not in k_map_parsed:
                continue
            if gname not in group_data:
                print(f"Skip save {gname}: no training rows", file=sys.stderr)
                continue
            k_i = int(k_map_parsed[gname])
            if k_i < 1:
                raise ValueError(f"{gname}: K must be >= 1, got {k_i}")
            Xm = group_data[gname]
            if Xm.shape[0] < k_i:
                raise ValueError(
                    f"{gname}: need at least K={k_i} rows for GMM, have {Xm.shape[0]} — "
                    "use Statcast parquet with enough pitches in this group"
                )
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xm)
            gmm = GaussianMixture(
                n_components=k_i,
                covariance_type=args.covariance_type,
                random_state=args.seed,
                reg_covar=args.reg_covar,
                max_iter=400,
                n_init=n_init_fit,
            ).fit(Xs)
            fname = f"pitch_group_{gname}_k{k_i}.joblib"
            path = args.out_dir / fname
            payload = {
                "scaler": scaler,
                "gmm": gmm,
                "feature_names": list(NUMERIC_CONTEXT_FEATURES),
                "pitch_group": gname,
                "k": k_i,
                "covariance_type": args.covariance_type,
                "n_train_rows": int(Xm.shape[0]),
            }
            joblib.dump(payload, path)
            saved_manifest[gname] = {"k": k_i, "path": str(path), "n_train_rows": int(Xm.shape[0])}
            print(f"Saved {path}")
        report["saved_models"] = saved_manifest

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_json = args.out_dir / "pitch_groups_gmm_report.json"
    out_csv = args.out_dir / "pitch_groups_gmm_aic_bic.csv"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(flat).to_csv(out_csv, index=False)

    if saved_manifest:
        man_path = args.out_dir / "pitch_groups_fitted_k_manifest.json"
        man_path.write_text(json.dumps(saved_manifest, indent=2), encoding="utf-8")
        print(f"Wrote {man_path}")

    print(f"Rows with mapped pitch groups (complete features): {n_used}")
    print(f"Date span: {report['game_date_min_observed']} .. {report['game_date_max_observed']}")
    print(f"Covariance: {args.covariance_type}\n")
    if flat:
        print(pd.DataFrame(flat).to_string(index=False))
    print(f"\nWrote {out_json}")
    if flat:
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
