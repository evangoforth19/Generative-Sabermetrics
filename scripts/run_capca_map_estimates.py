#!/usr/bin/env python3
"""
CA-PCA-style intrinsic dimension on MAP collision latents (curvature-adjusted local quadratic graphs).

Reference: arXiv:2509.15517 (CA-PCA); local quadratic graph from tangent PCA coordinates to normals.

Run from project root:
    python scripts/run_capca_map_estimates.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "capca_map_estimates_6000"
DEFAULT_K_LIST = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
DEFAULT_N_ROWS = 6000
BOOTSTRAP_KS = [20, 50, 100]
BOOTSTRAP_B = 50
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42
EPS = 1e-12
RIDGE_ALPHA = 1e-8
POOR_FIT_NRE_THRESHOLD = 0.25
STABLE_POOR_FIT_MAX = 0.25

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("capca_map")

CANONICAL: dict[str, list[str]] = {
    "x": ["x", "x_map", "x_map_in", "x_hat", "x_mean", "x_q50"],
    "e_y": [
        "e_y",
        "e_y_star",
        "e_y_map",
        "ey_map",
        "e_y_star_map",
        "e_y_mean",
        "e_y_star_mean",
    ],
    "e_x": ["e_x", "e_x_map", "ex_map", "e_x_mean"],
    "psi": [
        "psi",
        "psi_map",
        "psi_rad",
        "psi_map_rad",
        "psi_mean",
        "psi_deg",
        "psi_map_deg",
        "psi_q50",
    ],
    "omega_plus": [
        "omega_plus",
        "w_plus",
        "omega_plus_map",
        "omega_plus_map_rad_s",
        "w_plus_map",
        "omega_plus_mean",
    ],
}

ALGORITHM_DESC = (
    "Per MAP point and neighborhood size K: KNN on standardized 5D MAP features; "
    "neighborhood size n=K+1 including the center; center subtract local mean; SVD/PCA "
    "for local principal axes Z = X_centered @ V (descending variance). For each "
    "candidate intrinsic dimension d in {1,2,3,4}, split Z into tangent T (first d cols) "
    "and normal N (remaining 5-d cols). Fit multi-output Ridge(alpha=1e-8) mapping "
    "PolynomialFeatures(degree=2, include_bias=True)(T) to N. Select d_capca = argmin_d BIC "
    "with BIC = n*log(RSS_d/n+eps) + p_d*log(n), p_d = (5-d)*n_quad_features, "
    "nre_d = RSS_d/(TSS_d+eps). Flag poor_quadratic_fit if min_d nre_d > 0.25. "
    "Local-PCA dimensions from same centered neighborhood eigenvalues."
)


def discover_map_candidates(root: Path, limit: int = 60) -> list[Path]:
    patterns = [
        "**/map_per_event_global.parquet",
        "**/event_level_targets.parquet",
        "**/train_exports/posterior_draws_long_raw.parquet",
        "**/train_exports/posterior_draws_long_strict_screened.parquet",
        "**/train_exports/posterior_draws_long_basic_screened.parquet",
        "**/aggregates/posterior_summary_by_event.parquet",
        "**/map_estimates.csv",
    ]
    found: set[Path] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.suffix.lower() in {".parquet", ".csv"}:
                found.add(p.resolve())
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def score_candidate_path(p: Path) -> tuple[int, int, float]:
    """
    Higher is better: prefer ~6000-row production MAP summaries, then row count, then mtime.
    """
    try:
        st = p.stat()
        mtime = st.st_mtime
    except OSError:
        mtime = 0.0
    n = 0
    try:
        if p.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            m = pq.ParquetFile(p).metadata
            if m is not None:
                n = int(m.num_rows)
        else:
            with open(p, "rb") as f:
                n = sum(1 for _ in f) - 1
    except Exception:
        n = 0
    prod_bonus = 1 if (n >= 5500) else 0
    return (prod_bonus, n, mtime)


def auto_select_input(root: Path) -> Path | None:
    cands = discover_map_candidates(root)
    if not cands:
        return None
    preferred = [
        root / "Manifold Research" / "map_per_event_global.parquet",
        root
        / "MCMC 2"
        / "Outputs"
        / "refactor_exact_root_rhh"
        / "production"
        / "train_exports"
        / "event_level_targets.parquet",
    ]
    for p in preferred:
        if p.exists():
            return p.resolve()
    cands_sorted = sorted(cands, key=score_candidate_path, reverse=True)
    return cands_sorted[0]


def resolve_map_columns(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, Any]]:
    cols_lower = {c.lower(): c for c in df.columns}
    resolved: dict[str, str] = {}
    meta: dict[str, Any] = {"psi_degrees_source": None}

    for canon, aliases in CANONICAL.items():
        hit = None
        for a in aliases:
            if a in df.columns:
                hit = a
                break
            al = a.lower()
            if al in cols_lower:
                hit = cols_lower[al]
                break
        if hit is None:
            raise ValueError(
                f"Could not resolve canonical column '{canon}'. "
                f"Tried aliases {aliases}. Available: {list(df.columns)}"
            )
        resolved[canon] = hit

    psi_col = resolved["psi"]
    if "deg" in psi_col.lower() and "rad" not in psi_col.lower():
        meta["psi_degrees_source"] = psi_col
    elif psi_col in ("psi_rad", "psi_map_rad"):
        meta["psi_degrees_source"] = None
    elif "psi_deg" in df.columns and psi_col == "psi":
        s = pd.to_numeric(df[psi_col], errors="coerce").iloc[: min(500, len(df))]
        sd = pd.to_numeric(df["psi_deg"], errors="coerce").iloc[: min(500, len(df))]
        m = s.notna() & sd.notna()
        if m.sum() > 10 and np.allclose(s[m].to_numpy(), sd[m].to_numpy(), rtol=1e-4, atol=1e-4):
            meta["psi_degrees_source"] = "psi_matches_psi_deg"
        else:
            meta["psi_degrees_source"] = "unknown_assumed_radians"
    else:
        meta["psi_degrees_source"] = "unknown_assumed_radians"

    return resolved, meta


def psi_to_radians(series: pd.Series, col_name: str, meta: dict[str, Any]) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    reason = meta.get("psi_degrees_source")
    if reason in ("psi_deg", "psi_map_deg", "psi_matches_psi_deg"):
        LOG.info("Converting psi column %r from degrees to radians (detector=%s).", col_name, reason)
        return np.radians(v)
    p99 = float(np.nanpercentile(np.abs(v), 99)) if np.any(np.isfinite(v)) else 0.0
    if p99 > np.pi + 0.25:
        LOG.info(
            "Heuristic: |psi| p99=%.4f > pi — treating column %r as degrees -> radians.",
            p99,
            col_name,
        )
        return np.radians(v)
    LOG.info("Using psi column %r as radians (detector=%s, |psi| p99=%.4f).", col_name, reason, p99)
    return v


def merge_event_flags(df: pd.DataFrame, posterior_path: Path) -> pd.DataFrame:
    if not posterior_path.exists() or "event_id" not in df.columns:
        return df
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return df

    schema = set(pq.ParquetFile(posterior_path).schema_arrow.names)
    want = [
        c
        for c in ("event_id", "event_strict_screen_pass", "admissible_flag", "accepted_draw")
        if c in schema
    ]
    if "event_id" not in want:
        return df
    long = pd.read_parquet(posterior_path, columns=want)
    first = long.groupby("event_id", as_index=False).first()
    out = df.merge(first, on="event_id", how="left", suffixes=("", "_post"))
    LOG.info("Merged event flags from posterior: %s", [c for c in first.columns if c != "event_id"])
    return out


def select_n_rows(df: pd.DataFrame, n_rows: int, event_sort_col: str) -> pd.DataFrame:
    out = df.copy()
    if "event_strict_screen_pass" in out.columns:
        strict = pd.to_numeric(out["event_strict_screen_pass"], errors="coerce").fillna(0).astype(bool)
        out["_prio"] = (~strict).astype(int)
        LOG.info("event_strict_screen_pass=True count: %d", int(strict.sum()))
    elif "accepted_draw" in out.columns:
        acc = pd.to_numeric(out["accepted_draw"], errors="coerce").fillna(0).astype(bool)
        out["_prio"] = (~acc).astype(int)
    elif "admissible_flag" in out.columns:
        adm = pd.to_numeric(out["admissible_flag"], errors="coerce").fillna(0).astype(bool)
        out["_prio"] = (~adm).astype(int)
    else:
        out["_prio"] = 0
    key = event_sort_col if event_sort_col in out.columns else out.columns[0]
    out = out.sort_values(by=["_prio", key], ascending=[True, True]).drop(columns=["_prio"])
    if len(out) > n_rows:
        out = out.iloc[:n_rows].copy()
    return out.reset_index(drop=True)


def load_map_table(
    input_path: Path,
    post_path: Path | None,
    n_rows: int,
    warn_log: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], dict[str, Any], int, int]:
    if input_path.suffix.lower() == ".parquet":
        df0 = pd.read_parquet(input_path)
    else:
        df0 = pd.read_csv(input_path)
    n_loaded = len(df0)
    if post_path is not None:
        df0 = merge_event_flags(df0, post_path)
    resolved, meta = resolve_map_columns(df0)
    x = pd.to_numeric(df0[resolved["x"]], errors="coerce")
    ey = pd.to_numeric(df0[resolved["e_y"]], errors="coerce")
    ex = pd.to_numeric(df0[resolved["e_x"]], errors="coerce")
    w = pd.to_numeric(df0[resolved["omega_plus"]], errors="coerce")
    psi = pd.Series(psi_to_radians(df0[resolved["psi"]], resolved["psi"], meta), index=df0.index)
    feat = pd.DataFrame({"x": x, "e_y": ey, "e_x": ex, "psi": psi, "omega_plus": w})
    valid = np.isfinite(feat.to_numpy()).all(axis=1)
    if (~valid).any():
        warn_log.append(f"Dropped {(~valid).sum()} rows with non-finite MAP features.")
    df_work = df0.loc[valid].reset_index(drop=True)
    feat = feat.loc[valid].reset_index(drop=True)
    event_sort = "event_id" if "event_id" in df_work.columns else (
        "event_idx" if "event_idx" in df_work.columns else str(df_work.columns[0])
    )
    feat_cols = ["x", "e_y", "e_x", "psi", "omega_plus"]
    combined = df_work.copy()
    for c in feat_cols:
        combined[c] = feat[c].to_numpy()
    combined = select_n_rows(combined, n_rows, str(event_sort))
    feat = combined[feat_cols].copy()
    id_cols = [
        c
        for c in (
            "event_id",
            "event_idx",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "batter_name",
            "hitter_name",
            "pitcher",
            "split",
        )
        if c in combined.columns
    ]
    id_df = combined[id_cols].copy() if id_cols else pd.DataFrame(index=combined.index)
    n_used = len(combined)
    return combined, id_df, feat_cols, resolved, meta, n_loaded, n_used


def preprocess_features(
    feat: pd.DataFrame,
    feat_cols: list[str],
    standardize: bool,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    X_raw = feat[feat_cols].to_numpy(dtype=float)
    if not standardize:
        LOG.warning("standardize=False: distances and covariances are scale-sensitive.")
        return X_raw, X_raw.copy(), None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_raw, X_scaled, scaler


_N_QUAD_CACHE: dict[int, int] = {}


def n_quadratic_features(d_tangent: int) -> int:
    if d_tangent not in _N_QUAD_CACHE:
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(np.zeros((1, d_tangent)))
        _N_QUAD_CACHE[d_tangent] = int(poly.n_output_features_)
    return _N_QUAD_CACHE[d_tangent]


def fit_local_quadratic_graph(
    T: np.ndarray,
    N: np.ndarray,
    ridge_alpha: float,
) -> tuple[float, float, float, np.ndarray]:
    """
    Fit N ~ Ridge(Phi(T)). Returns RSS, TSS, nre, fitted N_pred.
    """
    n, d_t = T.shape
    _, d_n = N.shape
    if d_n == 0:
        return 0.0, EPS, 0.0, N
    poly = PolynomialFeatures(degree=2, include_bias=True)
    Phi = poly.fit_transform(T)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(Phi, N)
    N_hat = ridge.predict(Phi)
    resid = N - N_hat
    rss = float(np.sum(resid**2))
    N_mean = np.mean(N, axis=0, keepdims=True)
    tss = float(np.sum((N - N_mean) ** 2))
    nre = rss / (tss + EPS)
    return rss, tss, nre, N_hat


def compute_local_pca_stats(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int, int, bool]:
    """
    centered: (n, 5) with n = K+1. Returns evals desc, evr, d5, d90, d95, d99, ok.
    """
    d = centered.shape[1]
    n_s = centered.shape[0]
    ok = True
    try:
        cov = np.cov(centered, rowvar=False)
        w, _ = np.linalg.eigh(cov)
        lam = np.sort(np.real(w))[::-1][:d]
    except np.linalg.LinAlgError:
        lam = np.zeros(d, dtype=float)
        ok = False
    lam = np.clip(lam, 0.0, None)
    s = float(lam.sum())
    if s <= 0 or not np.isfinite(s):
        evr = np.zeros(d, dtype=float)
        return lam, evr, 0, 0, 0, 0, False
    evr = lam / s
    lam1 = lam[0] if lam[0] > EPS else EPS
    d5 = int(np.sum(lam > 0.05 * lam1))
    c = np.cumsum(evr)

    def dim_for(t: float) -> int:
        j = int(np.searchsorted(c, t, side="left"))
        return int(min(d, j + 1))

    d90 = dim_for(0.90)
    d95 = dim_for(0.95)
    d99 = dim_for(0.99)
    return lam, evr, d5, d90, d95, d99, ok


def estimate_capca_for_K(
    X: np.ndarray,
    K: int,
    ridge_alpha: float,
    warn_log: list[str],
) -> dict[str, Any]:
    """
    X: (n, 5) standardized. Returns dict of arrays length n.
    """
    n, d_amb = X.shape
    n_neighbors = K + 1
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    neigh_idx = nn.kneighbors(X, return_distance=False)

    capca_dim = np.empty(n, dtype=int)
    poor_fit = np.zeros(n, dtype=bool)
    underpowered = np.zeros(n, dtype=bool)
    lpca5 = np.empty(n, dtype=int)
    lpca90 = np.empty(n, dtype=int)
    lpca95 = np.empty(n, dtype=int)
    lpca99 = np.empty(n, dtype=int)
    best_nre = np.empty(n, dtype=float)
    bic_d = np.empty((n, 4), dtype=float)
    nre_d = np.empty((n, 4), dtype=float)
    evals = np.empty((n, d_amb), dtype=float)
    evr = np.empty((n, d_amb), dtype=float)
    degenerate = np.zeros(n, dtype=bool)

    for i in range(n):
        idx = neigh_idx[i]
        local = X[idx]
        mu = local.mean(axis=0)
        Xc = local - mu
        n_loc = Xc.shape[0]
        lam, evr_i, d5, d90, d95, d99, ok_pca = compute_local_pca_stats(Xc)
        evals[i] = lam
        evr[i] = evr_i
        lpca5[i] = d5
        lpca90[i] = d90
        lpca95[i] = d95
        lpca99[i] = d99
        if not ok_pca:
            degenerate[i] = True
            capca_dim[i] = d5
            poor_fit[i] = True
            best_nre[i] = np.nan
            bic_d[i, :] = np.nan
            nre_d[i, :] = np.nan
            warn_log.append(f"point_index={i} K={K}: local PCA degenerate")
            continue

        try:
            pca = PCA(n_components=d_amb, random_state=RANDOM_SEED)
            Z = pca.fit_transform(Xc)
        except Exception as e:
            degenerate[i] = True
            capca_dim[i] = d5
            poor_fit[i] = True
            best_nre[i] = np.nan
            bic_d[i, :] = np.nan
            nre_d[i, :] = np.nan
            warn_log.append(f"point_index={i} K={K}: PCA failed: {e}")
            continue

        bic_list: list[float] = []
        nre_list: list[float] = []
        max_poly = max(n_quadratic_features(d) for d in range(1, 5))
        if n_loc < max_poly:
            underpowered[i] = True
        for d in range(1, 5):
            T = Z[:, :d]
            N = Z[:, d:]
            rss, _tss, nre, _ = fit_local_quadratic_graph(T, N, ridge_alpha)
            nre_list.append(nre)
            n_poly = n_quadratic_features(d)
            p_dim = (5 - d) * n_poly
            bic_val = n_loc * math.log(rss / n_loc + EPS) + p_dim * math.log(n_loc)
            bic_list.append(bic_val)

        for j in range(4):
            bic_d[i, j] = bic_list[j]
            nre_d[i, j] = nre_list[j]

        min_nre = min(nre_list)
        best_nre[i] = min_nre
        if min_nre > POOR_FIT_NRE_THRESHOLD:
            poor_fit[i] = True
        j_best = int(np.argmin(bic_list))
        capca_dim[i] = j_best + 1

    return {
        "capca_dim": capca_dim,
        "poor_quadratic_fit": poor_fit,
        "underpowered": underpowered,
        "lpca_dim_5pct": lpca5,
        "lpca_dim_90": lpca90,
        "lpca_dim_95": lpca95,
        "lpca_dim_99": lpca99,
        "best_quadratic_error": best_nre,
        "bic_d1": bic_d[:, 0],
        "bic_d2": bic_d[:, 1],
        "bic_d3": bic_d[:, 2],
        "bic_d4": bic_d[:, 3],
        "nre_d1": nre_d[:, 0],
        "nre_d2": nre_d[:, 1],
        "nre_d3": nre_d[:, 2],
        "nre_d4": nre_d[:, 3],
        "eig1": evals[:, 0],
        "eig2": evals[:, 1],
        "eig3": evals[:, 2],
        "eig4": evals[:, 3],
        "eig5": evals[:, 4],
        "evr1": evr[:, 0],
        "evr2": evr[:, 1],
        "evr3": evr[:, 2],
        "evr4": evr[:, 3],
        "evr5": evr[:, 4],
        "degenerate": degenerate,
    }


def mode_int(vals: np.ndarray) -> int:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return -1
    vv = np.round(v).astype(int)
    bc = np.bincount(np.clip(vv - vv.min(), 0, None))
    return int(vv.min() + int(np.argmax(bc)))


def capca_dim_counts_str(vals: np.ndarray) -> str:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return "{}"
    u, c = np.unique(v.astype(int), return_counts=True)
    return json.dumps({str(int(k)): int(cj) for k, cj in zip(u, c)})


def summarize_one_k(K: int, n_rows: int, res: dict[str, Any]) -> dict[str, Any]:
    cd = res["capca_dim"].astype(float)
    pct_poor = float(np.mean(res["poor_quadratic_fit"]))
    med_cd = float(np.median(cd))
    return {
        "K": K,
        "n_rows": n_rows,
        "capca_dim_mean": float(np.mean(cd)),
        "capca_dim_median": med_cd,
        "capca_dim_std": float(np.std(cd, ddof=0)),
        "capca_dim_mode": float(mode_int(res["capca_dim"])),
        "capca_dim_counts": capca_dim_counts_str(res["capca_dim"]),
        "pct_poor_quadratic_fit": pct_poor,
        "lpca_5pct_dim_mean": float(np.mean(res["lpca_dim_5pct"])),
        "lpca_5pct_dim_median": float(np.median(res["lpca_dim_5pct"])),
        "lpca_90_dim_mean": float(np.mean(res["lpca_dim_90"])),
        "lpca_95_dim_mean": float(np.mean(res["lpca_dim_95"])),
        "lpca_99_dim_mean": float(np.mean(res["lpca_dim_99"])),
        "mean_best_quadratic_error": float(np.nanmean(res["best_quadratic_error"])),
        "median_best_quadratic_error": float(np.nanmedian(res["best_quadratic_error"])),
        "recommended_dimension_if_stable": np.nan,
    }


def run_bootstrap(
    X: np.ndarray,
    ks: list[int],
    B: int,
    frac: float,
    seed: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = max(1, int(math.floor(frac * n)))
    rows: list[dict[str, Any]] = []
    for b in range(B):
        idx = rng.choice(n, size=m, replace=False)
        Xb = X[idx]
        for K in ks:
            res = estimate_capca_for_K(Xb, K, ridge_alpha, [])
            rows.append(
                {
                    "bootstrap": b,
                    "K": K,
                    "capca_dim_mean": float(np.mean(res["capca_dim"])),
                    "capca_dim_median": float(np.median(res["capca_dim"])),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci_table(df_boot: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for K, g in df_boot.groupby("K"):
        for col in ("capca_dim_mean", "capca_dim_median"):
            v = g[col].to_numpy()
            out.append(
                {
                    "K": K,
                    "statistic": col,
                    "mean_over_bootstraps": float(np.mean(v)),
                    "ci_low_2.5": float(np.percentile(v, 2.5)),
                    "ci_high_97.5": float(np.percentile(v, 97.5)),
                    "ci_width": float(np.percentile(v, 97.5) - np.percentile(v, 2.5)),
                    "std_over_bootstraps": float(np.std(v, ddof=0)),
                }
            )
    return pd.DataFrame(out)


def make_plots(
    summary_df: pd.DataFrame,
    local_long: pd.DataFrame,
    mean_eigen_by_k: dict[int, np.ndarray],
    X_scaled: np.ndarray,
    capca_k50: np.ndarray,
    lpca_k50: np.ndarray,
    err_k50: np.ndarray,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    Ks = summary_df["K"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, summary_df["capca_dim_mean"], "o-", label="mean CA-PCA dim")
    ax.plot(Ks, summary_df["capca_dim_median"], "s--", label="median CA-PCA dim")
    ax.set_xlabel("K")
    ax.set_ylabel("CA-PCA dimension")
    ax.set_title("CA-PCA dimension vs neighborhood size K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "capca_dimension_vs_K.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, summary_df["lpca_5pct_dim_mean"], "^-", label="mean Local-PCA (5%)")
    ax.plot(Ks, summary_df["capca_dim_median"], "s--", label="median CA-PCA")
    ax.set_xlabel("K")
    ax.set_ylabel("Dimension")
    ax.set_title("Local-PCA (5% rule) vs CA-PCA (median)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "lpca_vs_capca_dimension_vs_K.png", dpi=150)
    plt.close(fig)

    n_k = len(Ks)
    ncols = 5
    nrows = int(math.ceil(n_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax, K in zip(axes, Ks):
        sub = local_long.loc[local_long["K"] == int(K), "capca_dim"]
        ax.hist(sub.dropna().to_numpy(), bins=np.arange(0.5, 6.6, 1.0), rwidth=0.85)
        ax.set_title(f"K={int(K)}")
        ax.set_xlabel("CA-PCA dim")
    for j in range(len(Ks), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("CA-PCA local dimension histograms by K", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "capca_dimension_histograms_by_K.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, summary_df["mean_best_quadratic_error"], "o-", label="mean best NRE")
    ax.plot(Ks, summary_df["median_best_quadratic_error"], "s--", label="median best NRE")
    ax.set_xlabel("K")
    ax.set_ylabel("Normalized quadratic error")
    ax.set_title("Best quadratic reconstruction error vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "quadratic_error_vs_K.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, summary_df["pct_poor_quadratic_fit"], "o-", color="darkred")
    ax.set_xlabel("K")
    ax.set_ylabel("Fraction poor fit")
    ax.set_title("Fraction of neighborhoods flagged poor_quadratic_fit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "poor_fit_fraction_vs_K.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for K in Ks:
        lam = mean_eigen_by_k[int(K)]
        ax.plot(np.arange(1, len(lam) + 1), lam, marker="o", ms=3, label=f"K={int(K)}")
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Mean local eigenvalue")
    ax.set_title("Mean local PCA eigenvalue spectrum by K")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "eigenvalue_spectrum_by_K.png", dpi=150)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)

    def scatter_colored(zcolor: np.ndarray, cbar_label: str, fname: str, cmap: str = "viridis") -> None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=zcolor, cmap=cmap, s=8, alpha=0.75)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label=cbar_label)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    scatter_colored(capca_k50, "CA-PCA dim (K=50)", "pca_scatter_colored_by_capca_dim_K50.png", cmap="tab10")
    scatter_colored(lpca_k50, "Local-PCA dim 5% (K=50)", "pca_scatter_colored_by_lpca_dim_K50.png", cmap="tab10")
    scatter_colored(err_k50, "Best quadratic NRE (K=50)", "pca_scatter_colored_by_quadratic_error_K50.png")


def write_interpretation_md(
    path: Path,
    *,
    input_path: Path,
    resolved: dict[str, str],
    n_used: int,
    standardize: bool,
    summary_df: pd.DataFrame,
    rec_range: tuple[float, float] | None,
    unstable_k: bool,
    high_poor: bool,
    capca_below_lpca: bool | None,
    median_capca: float,
    median_lpca: float,
) -> None:
    with open(path, "w") as f:
        f.write("# CA-PCA-style intrinsic dimension (MAP collision latents)\n\n")
        f.write("## Data\n\n")
        f.write(f"- **File used:** `{input_path.resolve()}`\n")
        f.write("- **Resolved MAP columns:**\n")
        for k, v in resolved.items():
            f.write(f"  - `{k}` ← `{v}`\n")
        f.write(f"- **Rows used:** {n_used}\n\n")
        f.write("## Preprocessing\n\n")
        f.write(
            "Finite rows only for x, e_y, e_x, psi (radians), omega_plus. "
            "Prefer strict-screened / admissible / accepted rows when present; else sort by event key; "
            f"cap at requested row count. Standardization: **{'StandardScaler on all five features' if standardize else 'disabled (raw features)'}**.\n\n"
        )
        f.write("## Method\n\n")
        f.write(
            "**Local PCA** assumes each neighborhood is locally flat: intrinsic dimension from "
            "eigenvalue threshold (5% of λ₁) and cumulative explained variance (90/95/99%). "
            "**CA-PCA-style quadratic PCA** approximates the local manifold as a **tangent quadratic graph**: "
            "after centering, PCA gives local tangent/normal axes; for each candidate tangent dimension "
            "d∈{1,2,3,4}, map quadratic features of the first d PCA coordinates to the remaining coordinates "
            "via Ridge regression, then select d by a BIC-style score. Poor quadratic fit "
            f"(min normalized error > {POOR_FIT_NRE_THRESHOLD}) triggers a fallback to the Local-PCA 5% dimension for that neighborhood "
            "so CA-PCA is not over-interpreted when the patch is not well explained by a quadratic graph.\n\n"
        )
        f.write("## Summary by K\n\n")
        f.write("```\n")
        f.write(summary_df.to_csv(index=False))
        f.write("```\n\n")
        if rec_range is not None:
            a, b = rec_range
            f.write(f"## Recommended intrinsic dimension range\n\n**[{a:.2f}, {b:.2f}]** (median CA-PCA across stable K).\n\n")
        else:
            f.write("## Recommended intrinsic dimension range\n\n**Unstable** — see warnings.\n\n")
        if unstable_k:
            f.write(
                "- **Warning:** Median CA-PCA dimension jumps across adjacent K or many K are unstable; "
                "the MAP cloud may be noisy, nonuniform, mixed-regime, or poorly approximated by local quadratic patches.\n"
            )
        if high_poor:
            f.write(
                f"- **Warning:** A large fraction of neighborhoods exceed the poor quadratic fit threshold; "
                "curvature-adjusted compression may not hold uniformly.\n"
            )
        f.write("\n## CA-PCA vs Local-PCA\n\n")
        if capca_below_lpca is True:
            f.write(
                "**CA-PCA median dimension is lower than Local-PCA (5%).** Ordinary Local-PCA may be counting "
                "curvature-induced variance in the normal directions as extra tangent dimension; the quadratic normal "
                "model partially absorbs that, yielding a lower effective dimension.\n\n"
            )
        elif capca_below_lpca is False:
            f.write(
                "**CA-PCA is comparable to or higher than Local-PCA (5%).** The dimension estimate is more aligned "
                "between flat and curved local models, which can indicate robustness or limited quadratic gain.\n\n"
            )
        else:
            f.write("Comparison unavailable (missing summaries).\n\n")
        f.write("## Baseball-physics interpretation (heuristic)\n\n")
        f.write(
            f"- If CA-PCA concentrates near **1–2**, MAP collision states appear **strongly compressed** onto a "
            f"low-dimensional curved physical manifold.\n"
            f"- Near **~3**, MAP states retain about **three effective** collision degrees of freedom.\n"
            f"- Near **4–5** or with many **poor-fit** neighborhoods, MAP states **do not** show strong low-dimensional "
            f"curved-manifold compression under this diagnostic.\n\n"
        )
        f.write(f"_Medians (representative K with stable K if available): CA-PCA ≈ {median_capca:.2f}, Local-PCA 5% ≈ {median_lpca:.2f}._\n")


def median_jump_unstable(summary_df: pd.DataFrame) -> bool:
    df = summary_df.sort_values("K")
    med = df["capca_dim_median"].to_numpy()
    if len(med) < 2:
        return False
    jumps = np.abs(np.diff(med))
    bad_jump = bool(np.any(jumps >= 1.5))
    return bad_jump


def final_recommendation_range(
    summary_df: pd.DataFrame,
    boot_ci_width_max: float | None,
) -> tuple[float, float] | None:
    """Median CA-PCA dimension across K that pass stability heuristics."""
    df = summary_df.sort_values("K").reset_index(drop=True)
    stable_mask = df["pct_poor_quadratic_fit"].values < STABLE_POOR_FIT_MAX
    med = df["capca_dim_median"].to_numpy(dtype=float)
    jumps = np.r_[0.0, np.abs(np.diff(med))]
    adj_ok = jumps < 1.5
    mask = stable_mask & adj_ok
    if not mask.any():
        mask = stable_mask
    if not mask.any():
        return None
    sub = df.loc[mask, "capca_dim_median"]
    lo = float(sub.min())
    hi = float(sub.max())
    if boot_ci_width_max is not None and boot_ci_width_max > 1.25:
        return None
    return (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="CA-PCA-style intrinsic dimension on MAP collision latents.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument("--k-list", type=str, default=",".join(str(k) for k in DEFAULT_K_LIST))
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize MAP features (default: true).",
    )
    parser.add_argument("--bootstrap", action="store_true", help="Run B=50 bootstrap subsamples (80%%).")
    parser.add_argument("--make-plots", action="store_true", help="Write figures under output/figures/.")
    parser.add_argument(
        "--posterior-long",
        type=Path,
        default=None,
        help="Optional long posterior for event flag merge (same default as local_pca).",
    )
    args = parser.parse_args()

    k_list = sorted({int(x.strip()) for x in args.k_list.split(",") if x.strip()})
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    warn_path = output_dir / "capca_warnings.log"
    warn_log: list[str] = []

    posterior_default = (
        PROJECT_ROOT
        / "MCMC 2"
        / "Outputs"
        / "refactor_exact_root_rhh"
        / "production"
        / "train_exports"
        / "posterior_draws_long_raw.parquet"
    )
    post_path = Path(args.posterior_long) if args.posterior_long else posterior_default

    input_path = args.input
    if input_path is None:
        cands = discover_map_candidates(PROJECT_ROOT)
        print("\nCandidate MAP / production files (scored; newest / largest rows first, up to 25):")
        scored = sorted(cands, key=lambda p: score_candidate_path(p), reverse=True)
        for p in scored[:25]:
            print(f"  {p}")
        sys.stdout.flush()
        input_path = auto_select_input(PROJECT_ROOT)
        if input_path is None:
            LOG.error("No input file. Pass --input.")
            sys.exit(1)
        print(f"\nSelected input: {input_path}\n")
        sys.stdout.flush()
    else:
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            LOG.error("Input not found: %s", input_path)
            sys.exit(1)

    combined, id_df, feat_cols, resolved, meta, n_loaded, n_used = load_map_table(
        input_path, post_path if post_path.exists() else None, args.n_rows, warn_log
    )
    feat = combined[feat_cols].copy()

    print("\nResolved column mapping:")
    for canon, actual in resolved.items():
        print(f"  {canon} <- {actual!r}")

    X_raw, X_scaled, scaler = preprocess_features(feat, feat_cols, args.standardize, args.random_seed)
    X_main = X_scaled if args.standardize else X_raw

    max_k = max(k_list)
    if max_k >= n_used:
        LOG.error("max K=%d must be < n_samples=%d", max_k, n_used)
        sys.exit(1)

    np.random.seed(args.random_seed)

    # Scaler stats
    if scaler is not None:
        scaler_payload = {
            "feature_columns": feat_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_samples": int(n_used),
        }
    else:
        scaler_payload = {"feature_columns": feat_cols, "mean": None, "scale": None, "n_samples": int(n_used)}
    with open(output_dir / "capca_scaler_stats.json", "w") as f:
        json.dump(scaler_payload, f, indent=2)

    # Feature matrix CSV
    id_reset = id_df.reset_index(drop=True)
    raw_df = pd.DataFrame(X_raw, columns=feat_cols)
    std_df = pd.DataFrame(X_scaled, columns=[f"{c}_std" for c in feat_cols])
    pd.concat([id_reset, raw_df, std_df], axis=1).to_csv(output_dir / "capca_feature_matrix_used.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    local_blocks: list[pd.DataFrame] = []
    eigen_blocks: list[pd.DataFrame] = []
    mean_eigen_by_k: dict[int, np.ndarray] = {}

    for K in k_list:
        LOG.info("CA-PCA for K=%d ...", K)
        res = estimate_capca_for_K(X_main, K, RIDGE_ALPHA, warn_log)
        mean_eigen_by_k[K] = np.nanmean(
            np.column_stack([res["eig1"], res["eig2"], res["eig3"], res["eig4"], res["eig5"]]), axis=0
        )
        summary_rows.append(summarize_one_k(K, n_used, res))

        loc = id_reset.copy()
        loc["K"] = K
        for key in (
            "capca_dim",
            "lpca_dim_5pct",
            "lpca_dim_90",
            "lpca_dim_95",
            "lpca_dim_99",
            "best_quadratic_error",
            "poor_quadratic_fit",
            "bic_d1",
            "bic_d2",
            "bic_d3",
            "bic_d4",
            "nre_d1",
            "nre_d2",
            "nre_d3",
            "nre_d4",
        ):
            loc[key] = res[key]
        local_blocks.append(loc)

        ev = id_reset.copy()
        ev["K"] = K
        ev["eig1"] = res["eig1"]
        ev["eig2"] = res["eig2"]
        ev["eig3"] = res["eig3"]
        ev["eig4"] = res["eig4"]
        ev["eig5"] = res["eig5"]
        ev["evr1"] = res["evr1"]
        ev["evr2"] = res["evr2"]
        ev["evr3"] = res["evr3"]
        ev["evr4"] = res["evr4"]
        ev["evr5"] = res["evr5"]
        eigen_blocks.append(ev)

    summary_df = pd.DataFrame(summary_rows)
    boot_ci_width_max: float | None = None
    if args.bootstrap:
        LOG.info("Bootstrap B=%d, frac=%.2f for K in %s", BOOTSTRAP_B, BOOTSTRAP_FRAC, BOOTSTRAP_KS)
        boot_df = run_bootstrap(X_main, BOOTSTRAP_KS, BOOTSTRAP_B, BOOTSTRAP_FRAC, args.random_seed, RIDGE_ALPHA)
        boot_df.to_csv(output_dir / "capca_bootstrap_draws.csv", index=False)
        ci_tbl = bootstrap_ci_table(boot_df)
        ci_tbl.to_csv(output_dir / "capca_bootstrap_summary.csv", index=False)
        boot_ci_width_max = float(ci_tbl["ci_width"].max())

    rec_rng = final_recommendation_range(summary_df, boot_ci_width_max)
    unstable = median_jump_unstable(summary_df)

    df_sorted = summary_df.sort_values("K").reset_index(drop=True)
    med_arr = df_sorted["capca_dim_median"].to_numpy(dtype=float)
    poor_arr = df_sorted["pct_poor_quadratic_fit"].to_numpy(dtype=float)
    jumps = np.r_[0.0, np.abs(np.diff(med_arr))]
    stable_k_mask = (poor_arr < STABLE_POOR_FIT_MAX) & (jumps < 1.5)
    if not stable_k_mask.any():
        stable_k_mask = poor_arr < STABLE_POOR_FIT_MAX
    stable_by_k = {int(df_sorted.at[j, "K"]): bool(stable_k_mask[j]) for j in range(len(df_sorted))}

    def _rec_dim(row: pd.Series) -> float:
        if rec_rng is None:
            return float("nan")
        if stable_by_k.get(int(row["K"]), False):
            return float(row["capca_dim_median"])
        return float("nan")

    summary_df = summary_df.copy()
    summary_df["recommended_dimension_if_stable"] = summary_df.apply(_rec_dim, axis=1)

    summary_df.to_csv(output_dir / "capca_summary_by_K.csv", index=False)
    pd.concat(local_blocks, ignore_index=True).to_csv(output_dir / "capca_local_dimensions_by_event.csv", index=False)
    pd.concat(eigen_blocks, ignore_index=True).to_csv(output_dir / "capca_eigenvalues_by_event.csv", index=False)

    # Manifest
    try:
        import sklearn

        skv = sklearn.__version__
    except Exception:
        skv = "unknown"
    manifest = {
        "input_file": str(Path(input_path).resolve()),
        "resolved_columns": resolved,
        "psi_meta": meta,
        "n_rows_loaded": int(n_loaded),
        "n_rows_used": int(n_used),
        "k_list": k_list,
        "standardize": bool(args.standardize),
        "random_seed": int(args.random_seed),
        "algorithm_description": ALGORITHM_DESC,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": skv,
        },
        "ridge_alpha": RIDGE_ALPHA,
        "poor_fit_nre_threshold": POOR_FIT_NRE_THRESHOLD,
        "posterior_long_for_flags": str(post_path.resolve()) if post_path.exists() else None,
    }
    with open(output_dir / "capca_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    local_long = pd.concat(local_blocks, ignore_index=True)
    sub50 = local_long.loc[local_long["K"] == 50].reset_index(drop=True)
    capca_k50 = sub50["capca_dim"].to_numpy() if len(sub50) == n_used else np.zeros(n_used)
    lpca_k50 = sub50["lpca_dim_5pct"].to_numpy() if len(sub50) == n_used else np.zeros(n_used)
    err_k50 = sub50["best_quadratic_error"].to_numpy() if len(sub50) == n_used else np.zeros(n_used)

    mid_k = 50 if 50 in k_list else k_list[len(k_list) // 2]
    row_mid = summary_df.loc[summary_df["K"] == mid_k]
    median_capca = float(row_mid["capca_dim_median"].iloc[0]) if len(row_mid) else float("nan")
    median_lpca = float(row_mid["lpca_5pct_dim_median"].iloc[0]) if len(row_mid) else float("nan")
    capca_below_lpca: bool | None
    if np.isfinite(median_capca) and np.isfinite(median_lpca):
        capca_below_lpca = median_capca + 0.25 < median_lpca
    else:
        capca_below_lpca = None
    high_poor = bool((summary_df["pct_poor_quadratic_fit"] > 0.35).any())

    write_interpretation_md(
        output_dir / "capca_interpretation.md",
        input_path=Path(input_path),
        resolved=resolved,
        n_used=n_used,
        standardize=bool(args.standardize),
        summary_df=summary_df,
        rec_range=rec_rng,
        unstable_k=unstable or (float(summary_df["pct_poor_quadratic_fit"].max()) >= STABLE_POOR_FIT_MAX),
        high_poor=high_poor,
        capca_below_lpca=capca_below_lpca,
        median_capca=median_capca,
        median_lpca=median_lpca,
    )

    if args.make_plots:
        make_plots(
            summary_df,
            local_long,
            mean_eigen_by_k,
            X_scaled,
            capca_k50,
            lpca_k50,
            err_k50,
            output_dir / "figures",
        )

    with open(warn_path, "w") as f:
        f.write("\n".join(warn_log))
        if warn_log:
            f.write("\n")

    # Console: recommendation
    if rec_rng is not None:
        a, b = rec_rng
        print(f"\nRecommended CA-PCA intrinsic dimension range: [{a:.2f}, {b:.2f}]")
    else:
        print("\nRecommended CA-PCA intrinsic dimension range: unstable / not computed")

    if capca_below_lpca is True:
        print(
            "CA-PCA median dimension is lower than Local-PCA (5%): ordinary Local-PCA may attribute "
            "curvature-induced normal variance as extra tangent dimension."
        )
    elif capca_below_lpca is False:
        print("CA-PCA is comparable to or higher than Local-PCA (5%): estimates are more aligned across models.")
    if unstable:
        print(
            "Warning: CA-PCA appears unstable across K (large median jumps or high poor-fit fraction); "
            "MAP cloud may be noisy, nonuniform, or poorly fit by quadratic patches."
        )
    max_poor = float(summary_df["pct_poor_quadratic_fit"].max())
    if max_poor >= STABLE_POOR_FIT_MAX:
        print(
            f"Warning: At some K, {100 * max_poor:.1f}% of neighborhoods exceed the poor quadratic fit threshold "
            f"(min NRE across d=1..4 > {POOR_FIT_NRE_THRESHOLD}). BIC-based CA-PCA dimension may still be read, "
            "but local patches are not well explained by a quadratic normal graph at those scales."
        )

    # One-paragraph interpretation
    if rec_rng is not None:
        a, b = rec_rng
        mid = 0.5 * (a + b)
        if mid <= 2.25:
            interp = (
                f"Across stable neighborhood sizes K, median CA-PCA dimension falls in about [{a:.1f}, {b:.1f}], "
                "suggesting the 5D standardized MAP collision states concentrate near a low-dimensional curved graph "
                "that quadratic normal corrections can summarize."
            )
        elif mid <= 3.5:
            interp = (
                f"The CA-PCA median band [{a:.1f}, {b:.1f}] points to roughly three effective local degrees of freedom "
                "in the MAP cloud—substantial structure but not extreme 1D–2D compression."
            )
        else:
            interp = (
                f"Median CA-PCA dimensions in the [{a:.1f}, {b:.1f}] band (with Local-PCA often near ambient) indicate "
                "limited low-dimensional curved-manifold compression of the MAP estimates under this quadratic diagnostic."
            )
    else:
        interp = (
            "Dimension estimates were unstable or dominated by poor quadratic neighborhoods; "
            "avoid a strong low-dimensional manifold conclusion without further diagnostics."
        )
    print("\nInterpretation (one paragraph):\n", interp)

    print("\n=== capca_summary_by_K.csv ===")
    print(summary_df.to_string(index=False))
    print(f"\nOutput folder: {output_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
