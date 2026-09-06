#!/usr/bin/env python3
"""
Kernel PCA (RBF): explained variance by number of components on MAP latent collision estimates.

Output: outputs/kernel_pca_variance/kernel_pca_variance_by_components.csv

Run from project root:
    python scripts/run_kernel_pca_variance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PREFERRED = [
    PROJECT_ROOT / "Manifold Research" / "map_per_event_global.parquet",
    PROJECT_ROOT
    / "MCMC 2"
    / "Outputs"
    / "refactor_exact_root_rhh"
    / "production"
    / "train_exports"
    / "event_level_targets.parquet",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kernel_pca_variance"
OUTPUT_CSV = "kernel_pca_variance_by_components.csv"

# Aliases (first match wins)
ALIASES: dict[str, list[str]] = {
    "x": ["x", "x_map", "x_map_in"],
    "e_y": ["e_y", "e_y_star", "e_y_map", "ey_map", "e_y_star_map"],
    "e_x": ["e_x", "e_x_map", "ex_map"],
    "psi": ["psi", "psi_map", "psi_map_rad", "psi_map_deg", "psi_rad", "psi_deg"],
    "omega_plus": [
        "omega_plus",
        "w_plus",
        "omega_plus_map",
        "omega_plus_map_rad_s",
        "w_plus_map",
    ],
}


def discover_input(root: Path) -> Path | None:
    patterns = [
        "**/map_per_event_global.parquet",
        "**/event_level_targets.parquet",
    ]
    found: list[Path] = []
    for pat in patterns:
        found.extend(p for p in root.glob(pat) if p.is_file() and p.suffix.lower() == ".parquet")
    ranked = sorted({p.resolve() for p in found}, key=lambda p: p.stat().st_mtime, reverse=True)
    for p in DEFAULT_INPUT_PREFERRED:
        if p.exists():
            return p
    return ranked[0] if ranked else None


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    lower = {c.lower(): c for c in df.columns}
    out: dict[str, str] = {}
    for canon, aliases in ALIASES.items():
        hit = None
        for a in aliases:
            if a in df.columns:
                hit = a
                break
            if a.lower() in lower:
                hit = lower[a.lower()]
                break
        if hit is None:
            raise ValueError(f"Could not resolve column '{canon}' from aliases {aliases}. Have: {list(df.columns)}")
        out[canon] = hit
    return out


def psi_to_radians(series: pd.Series, col: str, df: pd.DataFrame) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if "deg" in col.lower() and "rad" not in col.lower():
        return np.radians(v)
    if col == "psi" and "psi_deg" in df.columns:
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
    """gamma = 1 / median(squared pairwise Euclidean distance), off-diagonal."""
    # X already standardized; distances in feature space
    d2 = pairwise_distances(X, metric="sqeuclidean")
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(d2[iu]))
    if med <= 0 or not np.isfinite(med):
        raise ValueError("median pairwise squared distance is non-positive or non-finite")
    return 1.0 / med


def components_to_threshold(cum: np.ndarray, thresh: float) -> int | str:
    """1-based smallest j with cum[j-1] >= thresh."""
    if cum.size == 0:
        return "NA"
    if float(cum[-1]) < float(thresh) - 1e-12:
        return f">{cum.size}"
    idx = int(np.searchsorted(cum, thresh, side="left"))
    return idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel PCA variance by component (MAP latents).")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--n-rows", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = discover_input(PROJECT_ROOT)
    if input_path is None or not input_path.exists():
        print("No input file found. Pass --input.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(input_path) if input_path.suffix.lower() == ".parquet" else pd.read_csv(input_path)
    resolved = resolve_columns(df)
    print("Resolved columns:", resolved)

    x = pd.to_numeric(df[resolved["x"]], errors="coerce")
    ey = pd.to_numeric(df[resolved["e_y"]], errors="coerce")
    ex = pd.to_numeric(df[resolved["e_x"]], errors="coerce")
    psi = psi_to_radians(df[resolved["psi"]], resolved["psi"], df)
    w = pd.to_numeric(df[resolved["omega_plus"]], errors="coerce")

    feat = pd.DataFrame(
        {"x": x, "e_y": ey, "e_x": ex, "psi": psi, "omega_plus": w},
        index=df.index,
    )
    valid = np.isfinite(feat.to_numpy(dtype=float)).all(axis=1)
    feat = feat.loc[valid].reset_index(drop=True)
    if "event_id" in df.columns:
        ev = df.loc[valid, "event_id"].reset_index(drop=True)
        feat = feat.assign(_event_id=ev).sort_values("_event_id").drop(columns=["_event_id"])
    n = min(int(args.n_rows), len(feat))
    feat = feat.iloc[:n].copy()
    n_rows = len(feat)
    X = feat.to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_comp = min(20, n_rows - 1)
    if n_comp < 1:
        print("Need at least 2 finite rows.", file=sys.stderr)
        sys.exit(1)

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

    rows_out: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for gamma_label, gamma_val in gamma_specs:
        kpca = KernelPCA(
            n_components=n_comp,
            kernel="rbf",
            gamma=float(gamma_val),
            random_state=args.random_state,
            n_jobs=-1,
        )
        kpca.fit(Xs)
        raw_eig = np.asarray(kpca.eigenvalues_, dtype=float).ravel()
        pos = raw_eig[raw_eig > 0]
        if pos.size == 0:
            print(f"Warning: no positive eigenvalues for gamma={gamma_label} ({gamma_val})", file=sys.stderr)
            continue
        s = float(pos.sum())
        evr = pos / s
        cum = np.cumsum(evr)

        for j in range(len(pos)):
            rows_out.append(
                {
                    "kernel": "RBF",
                    "gamma_label": gamma_label,
                    "gamma_value": float(gamma_val),
                    "component": j + 1,
                    "eigenvalue": float(pos[j]),
                    "explained_variance_ratio": float(evr[j]),
                    "cumulative_explained_variance_ratio": float(cum[j]),
                }
            )

        summary_rows.append(
            {
                "gamma_label": gamma_label,
                "gamma_value": float(gamma_val),
                "components_for_80pct": components_to_threshold(cum, 0.80),
                "components_for_90pct": components_to_threshold(cum, 0.90),
                "components_for_95pct": components_to_threshold(cum, 0.95),
                "components_for_99pct": components_to_threshold(cum, 0.99),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / OUTPUT_CSV
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(rows_out)} rows).")

    summ = pd.DataFrame(summary_rows)
    print("\nComponents to reach cumulative EVR thresholds:\n")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
