#!/usr/bin/env python3
"""
Local-PCA intrinsic dimension on MAP estimates.

Modes (--feature-set):
  - latent_map (default): MAP collision latents x, e_y, e_x, psi, omega_plus.
  - mstar_exit_inputs: MAP posterior draw (argmax log_target) with m* / pitch / swing /
    plate metrics + hitter_meta constants; excludes exit-speed magnitudes.

Run from project root:
    python scripts/run_local_pca_map_estimates.py
    python scripts/run_local_pca_map_estimates.py --feature-set mstar_exit_inputs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "local_pca_map_estimates_6000"
DEFAULT_OUTPUT_MSTAR = PROJECT_ROOT / "outputs" / "local_pca_map_mstar_exit_inputs_6000"
DEFAULT_K_LIST = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
DEFAULT_N_ROWS = 6000
BOOTSTRAP_KS = [20, 50, 100]
BOOTSTRAP_B = 50
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("local_pca_map")

# MAP row on long posterior: m* / pitch / swing / plate context + merged hitter constants.
# Excludes exit-speed magnitudes (sensor draw and derived collision speed along batted-ball axis).
MSTAR_EXIT_NUMERIC = [
    "phi_star",
    "delta_star",
    "d_tilde",
    "a_tilde",
    "v_ss_tilde",
    "attack_direction_obs_star",
    "attack_angle_obs_star",
    "bat_speed_obs_star",
    "attack_direction_obs_deg",
    "attack_angle_obs_deg",
    "bat_speed_obs_mph",
    "launch_angle_obs_deg",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "spin_axis",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "balls",
    "strikes",
]
MSTAR_EXIT_CATEGORICAL = ["pitch_type", "stand", "p_throws"]
MSTAR_EXCLUDE_ALWAYS = {
    "exit_speed_obs_mph",
    "v_coll_mph",
    "v_coll_fps",
}
HITTER_META_CONST = [
    "L_in",
    "W_oz",
    "M_kg",
    "m_ball_kg",
    "r_ball_in",
    "alpha",
    "x_cm_fixed_in",
    "r_g_fixed_in",
    "I0_oz_in2",
    "Iz_oz_in2",
]

CANONICAL = {
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


def discover_map_candidates(root: Path, limit: int = 60) -> list[Path]:
    patterns = [
        "**/map_per_event_global.parquet",
        "**/event_level_targets.parquet",
        "**/train_exports/posterior_draws_long_raw.parquet",
        "**/train_exports/posterior_draws_long_strict_screened.parquet",
        "**/train_exports/posterior_draws_long_basic_screened.parquet",
        "**/aggregates/posterior_summary_by_event.parquet",
    ]
    found: set[Path] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.suffix.lower() in {".parquet", ".csv"}:
                found.add(p.resolve())
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


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
    # Plain "psi" in exports is often degrees (|psi| can exceed pi). Radian encodings rarely need p99 > pi.
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
    import pyarrow.parquet as pq

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


def load_mstar_map_frame(
    long_path: Path,
    hitter_meta_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """
    One MAP posterior draw per event_id (argmax log_target) on long exports.
    Features: transformed m* / swing / pitch / plate numerics + factorized categoricals
    + hitter_meta constants. Excludes exit-speed magnitudes.
    """
    import pyarrow.parquet as pq

    schema = set(pq.ParquetFile(long_path).schema_arrow.names)
    need = (
        ["event_id", "batter_name", "split", "log_target"]
        + [c for c in MSTAR_EXIT_NUMERIC if c in schema]
        + [c for c in MSTAR_EXIT_CATEGORICAL if c in schema]
        + [c for c in ("event_strict_screen_pass", "accepted_draw") if c in schema]
    )
    missing_num = [c for c in MSTAR_EXIT_NUMERIC if c not in schema]
    if missing_num:
        LOG.warning("Numeric columns missing from parquet (skipped): %s", missing_num)
    draws = pd.read_parquet(long_path, columns=need)
    hm = pd.read_csv(hitter_meta_path)
    hm["batter_name"] = hm["batter_name"].astype(str).str.lower().str.strip()
    draws["batter_name_norm"] = draws["batter_name"].astype(str).str.lower().str.strip()
    const_cols = [c for c in HITTER_META_CONST if c in hm.columns]
    hm_sub = hm[["batter_name"] + const_cols].drop_duplicates("batter_name")
    draws = draws.merge(hm_sub, left_on="batter_name_norm", right_on="batter_name", how="left", suffixes=("", "_hit"))
    if "batter_name_hit" in draws.columns:
        draws = draws.drop(columns=["batter_name_hit"])
    draws = draws.drop(columns=["batter_name_norm"], errors="ignore")

    g = draws.loc[draws["log_target"].notna()].copy()
    idx = g.groupby("event_id", sort=False)["log_target"].idxmax()
    mp = draws.loc[idx].reset_index(drop=True)
    n_map_events = len(mp)

    feat_parts: list[pd.Series | pd.DataFrame] = []
    feat_names: list[str] = []
    for c in MSTAR_EXIT_NUMERIC:
        if c not in mp.columns or c in MSTAR_EXCLUDE_ALWAYS:
            continue
        feat_parts.append(pd.to_numeric(mp[c], errors="coerce"))
        feat_names.append(c)
    for c in const_cols:
        if c in mp.columns:
            feat_parts.append(pd.to_numeric(mp[c], errors="coerce"))
            if c not in feat_names:
                feat_names.append(c)
    for c in MSTAR_EXIT_CATEGORICAL:
        if c not in mp.columns:
            continue
        code_col = f"{c}_code"
        raw = mp[c].astype(str).fillna("")
        codes, _ = pd.factorize(raw, sort=True)
        feat_parts.append(pd.Series(codes.astype(float), index=mp.index, name=code_col))
        feat_names.append(code_col)

    feat = pd.concat(feat_parts, axis=1)
    feat.columns = feat_names
    # Drop columns that are entirely or almost non-finite on MAP rows (e.g. unused export fields).
    arr_probe = feat.to_numpy(dtype=float)
    ok_frac = np.mean(np.isfinite(arr_probe), axis=0)
    keep_mask = ok_frac >= 0.95
    dropped_low_cov: list[str] = []
    if not np.all(keep_mask):
        dropped_low_cov = [feat_names[j] for j in range(len(feat_names)) if not keep_mask[j]]
        LOG.warning("Dropping low-coverage MAP feature columns (finite frac < 0.95): %s", dropped_low_cov)
        feat = feat.loc[:, keep_mask].copy()
        feat_names = [feat_names[j] for j in range(len(feat_names)) if keep_mask[j]]
    valid = np.isfinite(feat.to_numpy(dtype=float)).all(axis=1)
    mp = mp.loc[valid].reset_index(drop=True)
    feat = feat.loc[valid].reset_index(drop=True)

    id_cols = [c for c in ("event_id", "batter_name", "split") if c in mp.columns]
    id_df = mp[id_cols].copy() if id_cols else pd.DataFrame(index=mp.index)

    meta: dict[str, Any] = {
        "long_parquet": str(long_path.resolve()),
        "hitter_meta": str(hitter_meta_path.resolve()),
        "feature_columns": list(feat_names),
        "n_map_events_before_finite_filter": int(n_map_events),
        "excluded_always": sorted(MSTAR_EXCLUDE_ALWAYS),
        "missing_numeric_from_schema": missing_num,
        "dropped_low_coverage_columns": dropped_low_cov,
    }
    LOG.info("m* exit-input feature dimension d=%d columns: %s", len(feat_names), feat_names)
    combined = pd.concat([id_df, feat], axis=1)
    return combined, id_df, feat_names, meta


def select_n_rows(df: pd.DataFrame, n_rows: int, event_sort_col: str) -> pd.DataFrame:
    out = df.copy()
    if "event_strict_screen_pass" in out.columns:
        strict = pd.to_numeric(out["event_strict_screen_pass"], errors="coerce").fillna(0).astype(bool)
        out["_prio"] = (~strict).astype(int)
        LOG.info("event_strict_screen_pass=True count: %d", int(strict.sum()))
    else:
        out["_prio"] = 0
    key = event_sort_col if event_sort_col in out.columns else out.columns[0]
    out = out.sort_values(by=["_prio", key], ascending=[True, True]).drop(columns=["_prio"])
    if len(out) > n_rows:
        out = out.iloc[:n_rows].copy()
    return out.reset_index(drop=True)


def local_pca_one_k(X: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    if K >= n:
        raise ValueError(f"K={K} must be < n_samples={n}")
    nn = NearestNeighbors(n_neighbors=K + 1, metric="euclidean")
    nn.fit(X)
    neigh_idx = nn.kneighbors(X, return_distance=False)

    dims_5 = np.empty(n, dtype=int)
    dims_90 = np.empty(n, dtype=int)
    dims_95 = np.empty(n, dtype=int)
    dims_99 = np.empty(n, dtype=int)
    evals_all = np.empty((n, d), dtype=float)
    evr_all = np.empty((n, d), dtype=float)

    for i in range(n):
        local = X[neigh_idx[i]]
        mu = local.mean(axis=0)
        centered = local - mu
        cov = np.cov(centered, rowvar=False)
        w, _ = np.linalg.eigh(cov)
        lam = np.sort(np.real(w))[::-1][:d]
        lam = np.clip(lam, 0.0, None)
        # K+1 neighborhood points ⇒ sample covariance rank ≤ min(d, K); tail eigenvalues are noise here.
        r_cap = int(min(d, K))
        lam_use = lam[:r_cap]
        s = float(lam_use.sum())
        if s <= 0 or not np.isfinite(s):
            evals_all[i] = lam
            evr_all[i] = 0.0
            dims_5[i] = dims_90[i] = dims_95[i] = dims_99[i] = 0
            continue
        evr_use = lam_use / s
        evals_all[i, :r_cap] = lam_use
        evals_all[i, r_cap:] = 0.0
        evr_all[i, :r_cap] = evr_use
        evr_all[i, r_cap:] = 0.0
        lam1 = lam_use[0] if lam_use[0] > 0 else np.finfo(float).tiny
        dims_5[i] = int(np.sum(lam_use > 0.05 * lam1))
        c = np.cumsum(evr_use)

        def dim_for(t: float) -> int:
            j = int(np.searchsorted(c, t, side="left"))
            return int(min(r_cap, j + 1))

        dims_90[i] = dim_for(0.90)
        dims_95[i] = dim_for(0.95)
        dims_99[i] = dim_for(0.99)

    return dims_5, dims_90, dims_95, dims_99, evals_all, evr_all


def summarize_k(
    K: int,
    dims_5: np.ndarray,
    dims_90: np.ndarray,
    dims_95: np.ndarray,
    dims_99: np.ndarray,
    evals: np.ndarray,
    evr: np.ndarray,
) -> dict[str, Any]:
    n, d = evals.shape
    freq = {str(k): int(np.sum(dims_5 == k)) for k in range(0, d + 1)}
    row: dict[str, Any] = {
        "K": K,
        "global_dim_mean_5pct": float(np.mean(dims_5)),
        "global_dim_median_5pct": float(np.median(dims_5)),
        "global_dim_std_5pct": float(np.std(dims_5, ddof=0)),
        "global_dim_mean_90": float(np.mean(dims_90)),
        "global_dim_mean_95": float(np.mean(dims_95)),
        "global_dim_mean_99": float(np.mean(dims_99)),
        "freq_local_dim_5pct_json": json.dumps(freq),
    }
    for j in range(d):
        row[f"mean_eigenvalue_j{j + 1}"] = float(np.nanmean(evals[:, j]))
        row[f"median_eigenvalue_j{j + 1}"] = float(np.nanmedian(evals[:, j]))
        row[f"mean_evr_j{j + 1}"] = float(np.nanmean(evr[:, j]))
        row[f"median_evr_j{j + 1}"] = float(np.nanmedian(evr[:, j]))
    cum = np.cumsum(evr, axis=1)
    for j in range(d):
        row[f"mean_cum_evr_through_j{j + 1}"] = float(np.nanmean(cum[:, j]))
    return row


def run_bootstrap(X: np.ndarray, ks: list[int], B: int, frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = max(1, int(math.floor(frac * n)))
    rows = []
    for b in range(B):
        idx = rng.choice(n, size=m, replace=False)
        Xb = X[idx]
        for K in ks:
            d5, _, _, _, _, _ = local_pca_one_k(Xb, K)
            rows.append({"bootstrap": b, "K": K, "global_dim_mean_5pct": float(np.mean(d5))})
    return pd.DataFrame(rows)


def bootstrap_ci(df_boot: pd.DataFrame) -> pd.DataFrame:
    out = []
    for K, g in df_boot.groupby("K"):
        v = g["global_dim_mean_5pct"].to_numpy()
        out.append(
            {
                "K": K,
                "mean_over_bootstraps": float(np.mean(v)),
                "ci_low_2.5": float(np.percentile(v, 2.5)),
                "ci_high_97.5": float(np.percentile(v, 97.5)),
                "std_over_bootstraps": float(np.std(v, ddof=0)),
            }
        )
    return pd.DataFrame(out)


def plot_all(
    summary: pd.DataFrame,
    local_wide: pd.DataFrame,
    mean_eigen_by_k: dict[int, np.ndarray],
    mean_evr_by_k: dict[int, np.ndarray],
    cum_evr_by_k: dict[int, np.ndarray],
    Xs: np.ndarray,
    dim50: np.ndarray,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    d_amb = int(Xs.shape[1])
    Ks = summary["K"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, summary["global_dim_mean_5pct"], "o-", label="mean local_dim (5% rule)")
    ax.plot(Ks, summary["global_dim_median_5pct"], "s--", label="median local_dim (5% rule)")
    ax.set_xlabel("K (neighbors)")
    ax.set_ylabel("Intrinsic dimension estimate")
    ax.set_title("Global Local-PCA dimension vs neighborhood size K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "global_dim_vs_K.png", dpi=150)
    plt.close(fig)

    n_k = len(Ks)
    ncols = 5
    nrows = int(math.ceil(n_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax, K in zip(axes, Ks):
        col = f"local_dim_5pct_K{int(K)}"
        if col in local_wide.columns:
            ax.hist(
                local_wide[col].dropna().to_numpy(),
                bins=np.arange(0.5, float(d_amb) + 1.6, 1.0),
                rwidth=0.85,
            )
        ax.set_title(f"K={int(K)}")
        ax.set_xlabel("local_dim_5pct")
    for j in range(len(Ks), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Histogram of local_dim (5% eigenvalue rule)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "hist_local_dim_5pct_by_K.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for K in Ks:
        lam = mean_eigen_by_k[int(K)]
        ax.plot(np.arange(1, len(lam) + 1), lam, marker="o", ms=3, label=f"K={int(K)}")
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Mean local eigenvalue")
    ax.set_title("Mean local eigenvalue spectrum by K")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_eigenvalue_spectrum_by_K.png", dpi=150)
    plt.close(fig)

    mat = np.column_stack([mean_evr_by_k[int(k)] for k in Ks])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([str(int(k)) for k in Ks])
    ax.set_xlabel("K")
    ax.set_ylabel("Component j")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"j={j+1}" for j in range(mat.shape[0])])
    ax.set_title("Mean local explained variance ratio by component and K")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_evr_heatmap_by_K.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for K in Ks:
        c = cum_evr_by_k[int(K)]
        ax.plot(np.arange(1, len(c) + 1), c, marker="o", ms=3, label=f"K={int(K)}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Mean cumulative explained variance")
    ax.set_title("Mean cumulative local explained variance by K")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_cumulative_evr_by_K.png", dpi=150)
    plt.close(fig)

    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(
        Z[:, 0], Z[:, 1], c=dim50, cmap="tab10", vmin=0.5, vmax=float(d_amb) + 0.5, s=8, alpha=0.75
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of standardized MAP features (color: local_dim_5pct, K=50)")
    fig.colorbar(sc, ax=ax, label="local_dim_5pct")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca2d_colored_local_dim_K50.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    sc = ax3.scatter(
        Z[:, 0], Z[:, 1], Z[:, 2], c=dim50, cmap="tab10", vmin=0.5, vmax=float(d_amb) + 0.5, s=6, alpha=0.7
    )
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("3D PCA (color: local_dim_5pct, K=50)")
    fig.colorbar(sc, ax=ax3, shrink=0.5, label="local_dim_5pct")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca3d_colored_local_dim_K50.png", dpi=150)
    plt.close(fig)


def run_local_pca_core(
    *,
    X_raw: np.ndarray,
    id_df: pd.DataFrame,
    feat_cols: list[str],
    output_dir: Path,
    k_list: list[int],
    args: argparse.Namespace,
    input_path: Path,
    post_path: Path | None,
    feature_mode: str,
    latent_resolved: dict[str, str] | None,
    latent_meta: dict[str, Any] | None,
    mstar_meta: dict[str, Any] | None,
) -> None:
    """Shared Local-PCA loop (standardized main; optional unstd + bootstrap)."""
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_used = X_raw.shape[0]
    d_amb = int(X_raw.shape[1])
    LOG.info("Using n=%d rows, ambient dimension d=%d.", n_used, d_amb)

    max_k = max(k_list)
    if max_k >= n_used:
        raise ValueError(f"max K={max_k} must be < n_samples={n_used}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw)
    with open(output_dir / "local_pca_scaler_stats.json", "w") as f:
        json.dump(
            {
                "feature_mode": feature_mode,
                "feature_columns": feat_cols,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "n_samples": int(n_used),
            },
            f,
            indent=2,
        )

    pd.concat([id_df.reset_index(drop=True), pd.DataFrame(X_raw, columns=feat_cols)], axis=1).to_csv(
        output_dir / "local_pca_feature_matrix_used.csv", index=False
    )

    summary_rows: list[dict[str, Any]] = []
    dim_blocks: list[pd.DataFrame] = []
    mean_eigen_by_k: dict[int, np.ndarray] = {}
    mean_evr_by_k: dict[int, np.ndarray] = {}
    cum_evr_by_k: dict[int, np.ndarray] = {}

    for K in k_list:
        LOG.info("Local-PCA for K=%d ...", K)
        d5, d90, d95, d99, evals, evr = local_pca_one_k(Xs, K)
        summary_rows.append(summarize_k(K, d5, d90, d95, d99, evals, evr))
        blk = id_df.reset_index(drop=True).copy()
        blk[f"local_dim_5pct_K{K}"] = d5
        blk[f"local_dim_90_K{K}"] = d90
        blk[f"local_dim_95_K{K}"] = d95
        blk[f"local_dim_99_K{K}"] = d99
        dim_blocks.append(blk)

        ev_df = id_df.reset_index(drop=True).copy()
        for j in range(evals.shape[1]):
            ev_df[f"lambda_j{j + 1}"] = evals[:, j]
            ev_df[f"evr_j{j + 1}"] = evr[:, j]
            ev_df[f"cum_evr_j{j + 1}"] = np.cumsum(evr, axis=1)[:, j]
        ev_df.to_csv(output_dir / f"local_pca_eigenvalues_by_event_K{K}.csv", index=False)

        mean_eigen_by_k[K] = np.nanmean(evals, axis=0)
        mean_evr_by_k[K] = np.nanmean(evr, axis=0)
        cum_evr_by_k[K] = np.nanmean(np.cumsum(evr, axis=1), axis=0)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "local_pca_summary_by_K.csv", index=False)

    local_wide = id_df.reset_index(drop=True).copy()
    for blk in dim_blocks:
        for c in blk.columns:
            if c not in local_wide.columns:
                local_wide[c] = blk[c]
    local_wide.to_csv(output_dir / "local_pca_local_dimensions_by_event.csv", index=False)

    dim50 = local_wide["local_dim_5pct_K50"].to_numpy() if "local_dim_5pct_K50" in local_wide.columns else np.zeros(n_used)

    plot_all(summary_df, local_wide, mean_eigen_by_k, mean_evr_by_k, cum_evr_by_k, Xs, dim50, fig_dir)

    if not args.skip_unstandardized:
        LOG.info("Unstandardized diagnostic summary ...")
        rows_u = []
        for K in k_list:
            d5, d90, d95, d99, ev, evr = local_pca_one_k(X_raw, K)
            rows_u.append(summarize_k(K, d5, d90, d95, d99, ev, evr))
        pd.DataFrame(rows_u).to_csv(output_dir / "local_pca_summary_by_K_unstandardized.csv", index=False)

    if not args.skip_bootstrap:
        LOG.info("Bootstrap B=%d, frac=%.2f ...", BOOTSTRAP_B, BOOTSTRAP_FRAC)
        boot = run_bootstrap(Xs, BOOTSTRAP_KS, BOOTSTRAP_B, BOOTSTRAP_FRAC, args.random_seed)
        boot.to_csv(output_dir / "local_pca_bootstrap_draws.csv", index=False)
        bootstrap_ci(boot).to_csv(output_dir / "local_pca_bootstrap_summary.csv", index=False)

    means = summary_df["global_dim_mean_5pct"].to_numpy()
    span = float(np.max(means) - np.min(means))
    unstable = span > 0.75

    manifest: dict[str, Any] = {
        "feature_mode": feature_mode,
        "input_path": str(Path(input_path).resolve()),
        "posterior_long": str(post_path.resolve()) if post_path and post_path.exists() else None,
        "n_rows_used": int(n_used),
        "ambient_dimension": d_amb,
        "k_list": k_list,
        "unstable_across_K": unstable,
        "global_dim_mean_5pct_minmax": [float(np.min(means)), float(np.max(means))],
    }
    if feature_mode == "latent_map" and latent_resolved is not None:
        manifest["resolved_columns"] = latent_resolved
        manifest["psi_meta"] = latent_meta
    if feature_mode == "mstar_exit_inputs" and mstar_meta is not None:
        manifest["mstar_feature_spec"] = mstar_meta

    with open(output_dir / "local_pca_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    rec_lo = float(np.percentile(means, 25))
    rec_hi = float(np.percentile(means, 75))
    interp_path = output_dir / "local_pca_interpretation.md"
    with open(interp_path, "w") as f:
        if feature_mode == "latent_map":
            f.write("# Local-PCA intrinsic dimension (MAP collision latents)\n\n")
            f.write("## Data\n\n")
            f.write(f"- **File used:** `{input_path}`\n")
            if post_path and post_path.exists():
                f.write(f"- **Flags merged from:** `{post_path}`\n")
            f.write("- **Resolved columns:**\n")
            assert latent_resolved is not None
            for k, v in latent_resolved.items():
                f.write(f"  - `{k}` ← `{v}`\n")
            f.write(f"- **Rows used:** {n_used}\n\n")
            f.write("## Features\n\n")
            f.write(
                "Variables: `x`, `e_y`, `e_x`, `psi` (radians), `omega_plus`. "
                "**StandardScaler** before KNN. Scaler: `local_pca_scaler_stats.json`.\n\n"
            )
            f.write("## Method\n\n")
            f.write(
                "For each MAP point and **K**: **K** nearest neighbors in Euclidean distance on standardized "
                f"**{d_amb}**D vectors; neighborhood = center + **K** neighbors (**K+1** points). Local **sample covariance**; "
                "eigenvalues λ₁≥…≥λ_d. **local_dim_5pct** = count with λⱼ > 0.05 λ₁. "
                "**local_dim_90/95/99** = smallest m with cumulative explained variance ≥ 0.90/0.95/0.99. "
                "**Global** = mean of local_dim_5pct over events. "
                "Only the leading **min(d, K)** eigenvalues of each local covariance are used (rank bound for **K+1** points in **d**D), "
                "so numerical tail noise does not inflate dimension.\n\n"
            )
        else:
            f.write("# Local-PCA intrinsic dimension (MAP draw: m* + pitch/swing + hitter constants)\n\n")
            f.write("## Data\n\n")
            f.write(
                f"- **Long posterior (MAP = argmax log_target per event):** `{input_path}`\n"
                f"- **Hitter constants merged from:** `{mstar_meta.get('hitter_meta', '')}`\n"
            )
            f.write(f"- **Rows used:** {n_used}\n")
            f.write(f"- **Ambient feature dimension:** {d_amb}\n\n")
            f.write("## Features (exit-kinematics inputs, excluding exit speed magnitudes)\n\n")
            f.write(
                "Per MAP posterior draw: transformed measurement layer (`phi_star`, `delta_star`, `d_tilde`, "
                "`a_tilde`, `v_ss_tilde`, `attack_*_obs_star`, `bat_speed_obs_star`), observed attack/bat "
                "angles and speeds **except** `exit_speed_obs_mph`, pitch kinematics (`vx0`…`az`), plate/spin, "
                "`balls`/`strikes`, factorized `pitch_type` / `stand` / `p_throws`, plus hitter_meta constants "
                "(bat length, mass, radii, MOI, …). **Excluded:** `exit_speed_obs_mph`, `v_coll_mph`, `v_coll_fps`. "
                "Columns with sparse finite values on MAP rows may be auto-dropped before analysis.\n\n"
            )
            f.write("**Column list:** `" + "`, `".join(feat_cols) + "`\n\n")
            f.write("## Method\n\n")
            f.write(
                "Same Local-PCA neighborhood covariance as the latent MAP run: standardized Euclidean KNN, "
                f"local covariance of **{d_amb}**D vectors, 5% eigenvalue threshold for local dimension, "
                "global mean over events. Only **min(d, K)** leading eigenvalues per neighborhood are used.\n\n"
            )

        f.write("## Global dimension vs K\n\n```\n")
        f.write(summary_df.to_csv(index=False))
        f.write("```\n\n## Recommended range (heuristic)\n\n")
        f.write(
            f"- Mean local_dim (5% rule) across K: **{means.min():.3f}–{means.max():.3f}**. "
            f"Quartile band of the K-curve: **~{rec_lo:.2f}–{rec_hi:.2f}**.\n"
        )
        if unstable:
            f.write("- **Warning:** Large swing across K (range > 0.75); interpret with bootstrap CIs.\n")
        f.write("\n## Caveats\n\n")
        f.write(f"- Ambient dimension is **{d_amb}**; local_dim estimates lie in **[1, {d_amb}]**.\n")
        f.write("- Sensitive to **K**, curvature, noise, sampling; standardization changes neighborhoods.\n")
        if feature_mode == "mstar_exit_inputs":
            f.write(
                "- Categoricals are **factor codes** (not one-hot), so distances mix arbitrary label order.\n"
            )
        f.write("\n## Diagnostics\n\n")
        f.write("- Unstandardized: `local_pca_summary_by_K_unstandardized.csv`.\n")
        f.write("- Bootstrap: `local_pca_bootstrap_summary.csv` (K ∈ {20,50,100}).\n")

    LOG.info("Done. Interpretation: %s", interp_path)
    print("\n=== local_pca_summary_by_K.csv ===")
    print(summary_df.to_string(index=False))
    print(f"\nOutput folder: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-PCA intrinsic dimension on MAP estimates.")
    parser.add_argument(
        "--feature-set",
        choices=("latent_map", "mstar_exit_inputs"),
        default="latent_map",
        help="latent_map: MAP collision latents. mstar_exit_inputs: MAP draw m* + pitch/swing + hitter constants.",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--posterior-long",
        type=Path,
        default=None,
        help="Long posterior parquet (MAP source for m* mode; flag merge for latent mode).",
    )
    parser.add_argument("--hitter-meta", type=Path, default=PROJECT_ROOT / "MCMC 2" / "hitter_meta.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument("--k-list", type=str, default=",".join(str(k) for k in DEFAULT_K_LIST))
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-unstandardized", action="store_true")
    args = parser.parse_args()

    k_list = sorted({int(x.strip()) for x in args.k_list.split(",") if x.strip()})

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

    if args.feature_set == "mstar_exit_inputs":
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_MSTAR
        long_path = Path(args.input) if args.input else posterior_default
        if not long_path.exists():
            LOG.error("Long posterior not found: %s", long_path)
            sys.exit(1)
        LOG.info("m* exit-input mode: MAP rows from %s", long_path)
        combined, id_df, feat_cols, mstar_meta = load_mstar_map_frame(long_path, Path(args.hitter_meta))
        event_sort = "event_id" if "event_id" in combined.columns else str(combined.columns[0])
        combined = select_n_rows(combined, args.n_rows, str(event_sort))
        feat = combined[feat_cols].copy()
        id_df = combined[[c for c in id_df.columns if c in combined.columns]].copy()
        X_raw = feat.to_numpy(dtype=float)
        run_local_pca_core(
            X_raw=X_raw,
            id_df=id_df,
            feat_cols=feat_cols,
            output_dir=output_dir,
            k_list=k_list,
            args=args,
            input_path=long_path,
            post_path=post_path if post_path.exists() else None,
            feature_mode="mstar_exit_inputs",
            latent_resolved=None,
            latent_meta=None,
            mstar_meta=mstar_meta,
        )
        return

    # --- latent MAP (per-event table) ---
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    input_path = args.input
    if input_path is None:
        cands = discover_map_candidates(PROJECT_ROOT)
        LOG.info("Auto-discover candidates (newest first, showing up to 25):")
        for p in cands[:25]:
            print(f"  {p}")
        preferred = [
            PROJECT_ROOT / "Manifold Research" / "map_per_event_global.parquet",
            PROJECT_ROOT
            / "MCMC 2"
            / "Outputs"
            / "refactor_exact_root_rhh"
            / "production"
            / "train_exports"
            / "event_level_targets.parquet",
        ]
        input_path = next((p for p in preferred if p.exists()), cands[0] if cands else None)
        if input_path is None:
            LOG.error("No input file. Pass --input.")
            sys.exit(1)
        LOG.info("Selected input: %s", input_path)

    if input_path.suffix.lower() == ".parquet":
        df0 = pd.read_parquet(input_path)
    else:
        df0 = pd.read_csv(input_path)

    df0 = merge_event_flags(df0, post_path)
    resolved, meta = resolve_map_columns(df0)
    LOG.info("Resolved column mapping:")
    for canon, actual in resolved.items():
        print(f"  {canon} <- {actual!r}")

    x = pd.to_numeric(df0[resolved["x"]], errors="coerce")
    ey = pd.to_numeric(df0[resolved["e_y"]], errors="coerce")
    ex = pd.to_numeric(df0[resolved["e_x"]], errors="coerce")
    w = pd.to_numeric(df0[resolved["omega_plus"]], errors="coerce")
    psi = pd.Series(psi_to_radians(df0[resolved["psi"]], resolved["psi"], meta), index=df0.index)

    feat = pd.DataFrame({"x": x, "e_y": ey, "e_x": ex, "psi": psi, "omega_plus": w})
    valid = np.isfinite(feat.to_numpy()).all(axis=1)
    df_work = df0.loc[valid].reset_index(drop=True)
    feat = feat.loc[valid].reset_index(drop=True)

    event_sort = "event_id" if "event_id" in df_work.columns else (
        "event_idx" if "event_idx" in df_work.columns else str(df_work.columns[0])
    )
    feat_cols = ["x", "e_y", "e_x", "psi", "omega_plus"]
    combined = df_work.copy()
    for c in feat_cols:
        combined[c] = feat[c].to_numpy()

    combined = select_n_rows(combined, args.n_rows, str(event_sort))
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
            "split",
        )
        if c in combined.columns
    ]
    id_df = combined[id_cols].copy() if id_cols else pd.DataFrame(index=combined.index)
    X_raw = feat.to_numpy(dtype=float)

    run_local_pca_core(
        X_raw=X_raw,
        id_df=id_df,
        feat_cols=feat_cols,
        output_dir=output_dir,
        k_list=k_list,
        args=args,
        input_path=Path(input_path),
        post_path=post_path if post_path.exists() else None,
        feature_mode="latent_map",
        latent_resolved=resolved,
        latent_meta=meta,
        mstar_meta=None,
    )


if __name__ == "__main__":
    main()
