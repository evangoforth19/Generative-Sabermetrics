#!/usr/bin/env python3
"""
CA-PCA + Kernel PCA on an 8D MAP feature set:
  [incoming pitch speed, incoming pitch spin (w- proxy), bat speed, x, psi, e_y, e_x, w+]

Data source:
  Production posterior long parquet (train_exports/posterior_draws_long_raw.parquet).
  We keep splits train/test only, select one MAP row per event (argmax log_target),
  and keep up to 6000 finite rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "MCMC 2"
    / "Outputs"
    / "refactor_exact_root_rhh"
    / "production"
    / "train_exports"
    / "posterior_draws_long_raw.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "capca_kernel_pca_collision_plus_pitch_6000"
DEFAULT_K_LIST = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
EPS = 1e-12


def psi_to_radians(series: pd.Series, col_name: str, df: pd.DataFrame) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if "deg" in col_name.lower() and "rad" not in col_name.lower():
        return np.radians(v)
    if col_name == "psi" and "psi_deg" in df.columns:
        s = pd.to_numeric(series, errors="coerce").iloc[: min(500, len(df))]
        sd = pd.to_numeric(df["psi_deg"], errors="coerce").iloc[: min(500, len(df))]
        m = s.notna() & sd.notna()
        if m.sum() > 10 and np.allclose(s[m].to_numpy(), sd[m].to_numpy(), rtol=1e-4, atol=1e-4):
            return np.radians(v)
    p99 = float(np.nanpercentile(np.abs(v), 99)) if np.any(np.isfinite(v)) else 0.0
    if p99 > np.pi + 0.25:
        return np.radians(v)
    return v


def median_gamma_from_X(X: np.ndarray) -> float:
    d2 = pairwise_distances(X, metric="sqeuclidean")
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(d2[iu]))
    if med <= 0 or not np.isfinite(med):
        raise ValueError("median pairwise squared distance is non-positive or non-finite")
    return 1.0 / med


def components_to_threshold(cum: np.ndarray, thresh: float) -> int | str:
    if cum.size == 0:
        return "NA"
    if float(cum[-1]) < float(thresh) - 1e-12:
        return f">{cum.size}"
    idx = int(np.searchsorted(cum, thresh, side="left"))
    return idx + 1


def build_map_table(input_path: Path, n_rows: int) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(input_path) if input_path.suffix.lower() == ".parquet" else pd.read_csv(input_path)
    need = [
        "event_id",
        "split",
        "log_target",
        "vx0",
        "vy0",
        "vz0",
        "release_spin_rate",
        "bat_speed_obs_mph",
        "x",
        "psi",
        "e_y_star",
        "e_x",
        "omega_plus",
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in input: {miss}")
    df = df[need].copy()
    df = df[df["split"].astype(str).str.lower().isin(["train", "test"])].copy()
    df = df.sort_values(["event_id", "log_target"], ascending=[True, False]).drop_duplicates("event_id", keep="first")

    vx = pd.to_numeric(df["vx0"], errors="coerce")
    vy = pd.to_numeric(df["vy0"], errors="coerce")
    vz = pd.to_numeric(df["vz0"], errors="coerce")
    # vx0/vy0/vz0 are in ft/s in these production exports.
    speed_mph = np.sqrt(vx**2 + vy**2 + vz**2) * 0.681818
    psi_rad = psi_to_radians(df["psi"], "psi", df)

    out = pd.DataFrame(
        {
            "event_id": df["event_id"].to_numpy(),
            "split": df["split"].to_numpy(),
            "log_target": pd.to_numeric(df["log_target"], errors="coerce").to_numpy(dtype=float),
            "incoming_pitch_speed_mph": speed_mph.to_numpy(dtype=float),
            "incoming_pitch_spin_w_minus": pd.to_numeric(df["release_spin_rate"], errors="coerce").to_numpy(dtype=float),
            "bat_speed_mph": pd.to_numeric(df["bat_speed_obs_mph"], errors="coerce").to_numpy(dtype=float),
            "x": pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float),
            "psi": psi_rad,
            "e_y": pd.to_numeric(df["e_y_star"], errors="coerce").to_numpy(dtype=float),
            "e_x": pd.to_numeric(df["e_x"], errors="coerce").to_numpy(dtype=float),
            "omega_plus": pd.to_numeric(df["omega_plus"], errors="coerce").to_numpy(dtype=float),
        }
    )
    feat_cols = [
        "incoming_pitch_speed_mph",
        "incoming_pitch_spin_w_minus",
        "bat_speed_mph",
        "x",
        "psi",
        "e_y",
        "e_x",
        "omega_plus",
    ]
    valid = np.isfinite(out[feat_cols].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[valid].sort_values("event_id").reset_index(drop=True)
    if len(out) > n_rows:
        out = out.iloc[:n_rows].copy()
    return out, feat_cols


def fit_local_quadratic_graph(T: np.ndarray, N: np.ndarray, ridge_alpha: float) -> tuple[float, float]:
    if N.shape[1] == 0:
        return 0.0, 0.0
    poly = PolynomialFeatures(degree=2, include_bias=True)
    Phi = poly.fit_transform(T)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(Phi, N)
    N_hat = ridge.predict(Phi)
    rss = float(np.sum((N - N_hat) ** 2))
    nre = rss / (float(np.sum((N - np.mean(N, axis=0, keepdims=True)) ** 2)) + EPS)
    return rss, nre


def estimate_capca_for_k(X: np.ndarray, k: int, ridge_alpha: float) -> dict[str, np.ndarray]:
    n, d_amb = X.shape
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    neigh_idx = nn.kneighbors(X, return_distance=False)
    capca_dim = np.zeros(n, dtype=int)
    poor_fit = np.zeros(n, dtype=bool)
    best_nre = np.zeros(n, dtype=float)
    eigvals = np.zeros((n, d_amb), dtype=float)

    for i in range(n):
        local = X[neigh_idx[i]]
        Xc = local - np.mean(local, axis=0, keepdims=True)
        cov = np.cov(Xc, rowvar=False)
        w, _ = np.linalg.eigh(cov)
        lam = np.sort(np.real(w))[::-1]
        lam = np.clip(lam, 0.0, None)
        eigvals[i] = lam
        pca = PCA(n_components=d_amb, random_state=42)
        Z = pca.fit_transform(Xc)
        bic = []
        nre_list = []
        n_loc = Xc.shape[0]
        for d in range(1, d_amb):
            T = Z[:, :d]
            N = Z[:, d:]
            rss, nre = fit_local_quadratic_graph(T, N, ridge_alpha)
            n_poly = PolynomialFeatures(degree=2, include_bias=True).fit(np.zeros((1, d))).n_output_features_
            p_dim = (d_amb - d) * n_poly
            bic_val = n_loc * math.log(rss / n_loc + EPS) + p_dim * math.log(n_loc)
            bic.append(bic_val)
            nre_list.append(nre)
        j_best = int(np.argmin(bic))
        capca_dim[i] = j_best + 1
        best_nre[i] = float(np.min(nre_list))
        poor_fit[i] = best_nre[i] > 0.25
    return {"capca_dim": capca_dim, "best_nre": best_nre, "poor_fit": poor_fit, "eigvals": eigvals}


def main() -> None:
    parser = argparse.ArgumentParser(description="CA-PCA + Kernel PCA on 8D MAP feature set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-rows", type=int, default=6000)
    parser.add_argument("--k-list", type=str, default=",".join(str(k) for k in DEFAULT_K_LIST))
    parser.add_argument("--ridge-alpha", type=float, default=1e-8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    capca_dir = out_dir / "capca"
    kpca_dir = out_dir / "kernel_pca"
    capca_dir.mkdir(parents=True, exist_ok=True)
    kpca_dir.mkdir(parents=True, exist_ok=True)

    table, feat_cols = build_map_table(input_path, args.n_rows)
    X = table[feat_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pd.concat([table[["event_id", "split", "log_target"]], pd.DataFrame(X, columns=feat_cols)], axis=1).to_csv(
        out_dir / "feature_matrix_used.csv", index=False
    )

    # Kernel PCA
    n_comp = min(20, len(Xs) - 1)
    gamma_median = median_gamma_from_X(Xs)
    gamma_specs: list[tuple[str, float]] = [
        ("median", gamma_median),
        ("0.01", 0.01),
        ("0.05", 0.05),
        ("0.1", 0.1),
        ("0.5", 0.5),
        ("1.0", 1.0),
        ("2.0", 2.0),
        ("5.0", 5.0),
    ]
    kpca_rows: list[dict[str, Any]] = []
    kpca_summary: list[dict[str, Any]] = []
    for label, gamma in gamma_specs:
        kpca = KernelPCA(n_components=n_comp, kernel="rbf", gamma=float(gamma), random_state=args.random_state, n_jobs=-1)
        kpca.fit(Xs)
        eig = np.asarray(kpca.eigenvalues_, dtype=float).ravel()
        pos = eig[eig > 0]
        if pos.size == 0:
            continue
        evr = pos / float(pos.sum())
        cum = np.cumsum(evr)
        for j in range(len(pos)):
            kpca_rows.append(
                {
                    "gamma_label": label,
                    "gamma_value": float(gamma),
                    "component": j + 1,
                    "eigenvalue": float(pos[j]),
                    "explained_variance_ratio": float(evr[j]),
                    "cumulative_explained_variance_ratio": float(cum[j]),
                }
            )
        kpca_summary.append(
            {
                "gamma_label": label,
                "gamma_value": float(gamma),
                "components_for_80pct": components_to_threshold(cum, 0.80),
                "components_for_90pct": components_to_threshold(cum, 0.90),
                "components_for_95pct": components_to_threshold(cum, 0.95),
                "components_for_99pct": components_to_threshold(cum, 0.99),
            }
        )
    pd.DataFrame(kpca_rows).to_csv(kpca_dir / "kernel_pca_variance_by_components.csv", index=False)
    pd.DataFrame(kpca_summary).to_csv(kpca_dir / "kernel_pca_components_for_thresholds.csv", index=False)

    # CA-PCA
    k_list = sorted({int(x.strip()) for x in args.k_list.split(",") if x.strip()})
    if max(k_list) >= len(Xs):
        raise ValueError(f"max K={max(k_list)} must be < n_samples={len(Xs)}")
    capca_summary_rows: list[dict[str, Any]] = []
    local_rows: list[pd.DataFrame] = []
    for k in k_list:
        res = estimate_capca_for_k(Xs, k, args.ridge_alpha)
        capca_summary_rows.append(
            {
                "K": k,
                "n_rows": int(len(Xs)),
                "capca_dim_mean": float(np.mean(res["capca_dim"])),
                "capca_dim_median": float(np.median(res["capca_dim"])),
                "capca_dim_std": float(np.std(res["capca_dim"])),
                "pct_poor_quadratic_fit": float(np.mean(res["poor_fit"])),
                "mean_best_quadratic_error": float(np.mean(res["best_nre"])),
            }
        )
        block = table[["event_id", "split"]].copy()
        block["K"] = k
        block["capca_dim"] = res["capca_dim"]
        block["best_quadratic_error"] = res["best_nre"]
        block["poor_quadratic_fit"] = res["poor_fit"]
        local_rows.append(block)
    pd.DataFrame(capca_summary_rows).to_csv(capca_dir / "capca_summary_by_K.csv", index=False)
    pd.concat(local_rows, ignore_index=True).to_csv(capca_dir / "capca_local_dimensions_by_event.csv", index=False)

    manifest = {
        "input_file": str(input_path.resolve()),
        "n_rows_used": int(len(table)),
        "splits_used": ["train", "test"],
        "features": feat_cols,
        "feature_notes": {
            "incoming_pitch_spin_w_minus": "Proxy uses release_spin_rate from production exports.",
            "incoming_pitch_speed_mph": "Computed as sqrt(vx0^2+vy0^2+vz0^2) * 0.681818 from ft/s to mph.",
            "e_y": "Mapped from e_y_star in long posterior exports.",
        },
        "k_list_capca": k_list,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir.resolve()}")
    print(f"Rows used: {len(table)}")
    print("Features:", feat_cols)
    print("\nKernel PCA thresholds:")
    print(pd.DataFrame(kpca_summary).to_string(index=False))
    print("\nCA-PCA summary:")
    print(pd.DataFrame(capca_summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
