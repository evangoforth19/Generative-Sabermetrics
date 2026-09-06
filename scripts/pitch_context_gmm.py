#!/usr/bin/env python3
"""
Fit separate Gaussian Mixture Models on pitch *context* features, split by pitcher handedness.

Features match sbi_forward_sim ``numeric_context_features`` (7D):
  z_count, plate_x, plate_z, release_spin_rate, spin_axis_sin, spin_axis_cos, release_speed

- ``z_count`` is built from balls/strikes via the same map as
  ``MCMC 2/sbi_forward_sim/src/feature_engineering.py`` if absent.
- ``spin_axis_*`` are built from Statcast ``spin_axis`` (degrees).

Outputs CSV + JSON with AIC and BIC vs K (per sklearn GaussianMixture on scaled data).

Example:
  python scripts/pitch_context_gmm.py \\
    --parquet data/statcast_pybaseball/statcast_all.parquet \\
    --date-start 2020-01-01 --date-end 2026-12-31 \\
    --k-min 1 --k-max 20

For full 2020–present coverage, build a large Statcast Parquet first, e.g.:
  python scripts/download_statcast_pybaseball.py --start 2020-01-01 --end 2025-12-31 \\
    --output-dir data/statcast_pybaseball
  (then --combine-only to produce statcast_all.parquet, or pass --parquet-glob on chunks)

Refit at chosen K and save for sampling (Z-scored space, then inverse-transform):
  python scripts/pitch_context_gmm.py ... --save-fitted-k 14

  # sample 1000 RHP pitch contexts in original units:
  import joblib, numpy as np
  p = joblib.load("outputs/pitch_context_gmm/pitch_context_gmm_rhp.joblib")
  z, comp = p["gmm"].sample(1000)
  X = p["scaler"].inverse_transform(z)  # columns: feature_names
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

# Same map as sbi_forward_sim feature_engineering.z_count_from_balls_strikes
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


def _parquet_read_columns() -> list[str]:
    return [
        "game_date",
        "p_throws",
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
    cols = _parquet_read_columns()
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p, columns=cols))
    return pd.concat(frames, ignore_index=True)


def prepare_design_matrix(df: pd.DataFrame):
    need_cols = ["game_date", "p_throws", "plate_x", "plate_z", "release_spin_rate", "release_speed"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Input missing columns: {miss}")

    d = df.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d[d["game_date"].notna()].copy()

    d["p_throws"] = d["p_throws"].astype(str).str.strip().str.upper()
    d = d[d["p_throws"].isin(["L", "R"])].copy()

    d = add_z_count(d)
    d = add_spin_trig(d)

    for c in NUMERIC_CONTEXT_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    before = len(d)
    d = d.dropna(subset=NUMERIC_CONTEXT_FEATURES)
    after = len(d)
    dropped_na = before - after

    X = d[NUMERIC_CONTEXT_FEATURES].to_numpy(dtype=np.float64)
    handed = d["p_throws"].to_numpy()
    dates = d["game_date"]
    return X, handed, dates, dropped_na, len(d)


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
    ap = argparse.ArgumentParser(description="GMM on pitch context features (LHP/RHP separate)")
    ap.add_argument("--parquet", type=Path, default=None, help="Single Statcast parquet")
    ap.add_argument(
        "--parquet-glob",
        type=str,
        default=None,
        help="Glob of parquet files (e.g. data/statcast_pybaseball/chunks/*.parquet)",
    )
    ap.add_argument("--date-start", type=str, default="2020-01-01")
    ap.add_argument("--date-end", type=str, default="2099-12-31")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows-per-hand", type=int, default=None, help="Subsample after filtering (per handedness)")
    ap.add_argument("--reg-covar", type=float, default=1e-4)
    ap.add_argument(
        "--covariance-type",
        choices=("full", "diag", "tied", "spherical"),
        default="diag",
        help="diag/tied/spherical are much faster than full on large N (full can be very slow).",
    )
    ap.add_argument("--n-init", type=int, default=2, help="Number of random initializations per K")
    ap.add_argument(
        "--save-fitted-k",
        type=int,
        default=None,
        help="If set, refit and save one GMM per handedness (joblib) at this K for sampling",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/pitch_context_gmm"))
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
    df = df[(pd.to_datetime(df["game_date"], errors="coerce") >= t0) & (pd.to_datetime(df["game_date"], errors="coerce") <= t1)].copy()

    rng = np.random.default_rng(args.seed)

    X, handed, dates, dropped_na, n_used = prepare_design_matrix(df)

    report = {
        "parquet_paths": [str(p) for p in paths],
        "date_start": args.date_start,
        "date_end": args.date_end,
        "game_date_min_observed": str(pd.to_datetime(dates).min().date()),
        "game_date_max_observed": str(pd.to_datetime(dates).max().date()),
        "n_rows_after_date_filter": int(len(df)),
        "n_rows_complete_features": int(n_used),
        "rows_dropped_missing_features": int(dropped_na),
        "features": NUMERIC_CONTEXT_FEATURES,
        "covariance_type": args.covariance_type,
        "n_init": args.n_init,
        "lhp": {},
        "rhp": {},
    }

    ks = range(args.k_min, args.k_max + 1)

    for label, mask_char in (("lhp", "L"), ("rhp", "R")):
        Xm = X[handed == mask_char]
        if args.max_rows_per_hand is not None and Xm.shape[0] > args.max_rows_per_hand:
            idx = rng.choice(Xm.shape[0], size=args.max_rows_per_hand, replace=False)
            Xm = Xm[idx]
        if Xm.shape[0] < 50:
            report[label]["error"] = f"Too few rows ({Xm.shape[0]}); need Statcast extract"
            continue
        rows = fit_gmm_grid(
            Xm,
            ks,
            seed=args.seed,
            reg_covar=args.reg_covar,
            covariance_type=args.covariance_type,
            n_init=args.n_init,
        )
        report[label]["n_rows"] = int(Xm.shape[0])
        report[label]["aic_bic_by_k"] = rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "pitch_context_gmm_report.json"
    out_csv = args.out_dir / "pitch_context_gmm_aic_bic.csv"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    flat = []
    for hand_key, title in (("lhp", "L"), ("rhp", "R")):
        block = report.get(hand_key, {})
        for row in block.get("aic_bic_by_k", []):
            flat.append({"handedness": title, **row})

    pd.DataFrame(flat).to_csv(out_csv, index=False)

    if args.save_fitted_k is not None:
        k_save = int(args.save_fitted_k)
        for label, mask_char, fname in (
            ("lhp", "L", "pitch_context_gmm_lhp.joblib"),
            ("rhp", "R", "pitch_context_gmm_rhp.joblib"),
        ):
            Xm = X[handed == mask_char]
            if args.max_rows_per_hand is not None and Xm.shape[0] > args.max_rows_per_hand:
                idx = rng.choice(Xm.shape[0], size=args.max_rows_per_hand, replace=False)
                Xm = Xm[idx]
            if Xm.shape[0] < 50:
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xm)
            gmm = GaussianMixture(
                n_components=k_save,
                covariance_type=args.covariance_type,
                random_state=args.seed,
                reg_covar=args.reg_covar,
                max_iter=300,
                n_init=max(args.n_init, 3),
            ).fit(Xs)
            payload = {
                "scaler": scaler,
                "gmm": gmm,
                "feature_names": list(NUMERIC_CONTEXT_FEATURES),
                "handedness": mask_char,
                "k": k_save,
                "covariance_type": args.covariance_type,
                "n_train_rows": Xm.shape[0],
            }
            joblib.dump(payload, args.out_dir / fname)
            print(f"Saved {args.out_dir / fname}")

    print(f"Rows LHP: {report['lhp'].get('n_rows', 'n/a')}  RHP: {report['rhp'].get('n_rows', 'n/a')}")
    print(f"Date span in data: {report['game_date_min_observed']} .. {report['game_date_max_observed']}")
    print(f"Covariance: {args.covariance_type}\n")
    if flat:
        pdf = pd.DataFrame(flat)
        print(pdf.to_string(index=False))
    print(f"\nWrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
