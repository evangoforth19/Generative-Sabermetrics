#!/usr/bin/env python3
"""
Nonlinear surrogate + SHAP for local intrinsic-subspace composition at K=10.

Targets:
  For each event i, compute local tangent-subspace composition c_i in original features:
    c_i[j] = sum_{r=1..d} v_{r,j}^2
  where v_{r,*} are top-d right singular vectors of the centered KNN neighborhood.

Datasets:
  - 5D latent MAP set: x, e_y, e_x, psi, omega_plus (d=3)
  - 8D full physics+latent set from feature_matrix_used.csv (d=6)

Model:
  - One HistGradientBoostingRegressor per composition target dimension.
  - SHAP TreeExplainer on each target model, aggregated via mean(|SHAP|).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "nonlinear_surrogate_shap_k10"
K_NEIGH = 10


def psi_to_radians(series: pd.Series) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    p99 = float(np.nanpercentile(np.abs(v), 99)) if np.any(np.isfinite(v)) else 0.0
    if p99 > np.pi + 0.25:
        return np.radians(v)
    return v


def local_subspace_composition(X: np.ndarray, d: int, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    n, p = X.shape
    C = np.zeros((n, p), dtype=float)
    for i in range(n):
        Z = X[idx[i]]
        Zc = Z - Z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(Zc, full_matrices=False)
        Vd = vt[:d, :]
        C[i] = np.sum(Vd * Vd, axis=0)
    return C


def fit_gbm_surrogates(
    X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray
) -> tuple[list[HistGradientBoostingRegressor], np.ndarray]:
    preds = np.zeros((X_test.shape[0], Y_train.shape[1]), dtype=float)
    models: list[HistGradientBoostingRegressor] = []
    for j in range(Y_train.shape[1]):
        m = HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=400,
            learning_rate=0.05,
            l2_regularization=1e-2,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42,
        )
        m.fit(X_train, Y_train[:, j])
        preds[:, j] = m.predict(X_test)
        models.append(m)
    return models, preds


def shap_importance(
    models: list[HistGradientBoostingRegressor],
    X_background: np.ndarray,
    X_eval: np.ndarray,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    global_accum = np.zeros(len(feature_cols), dtype=float)
    for j, model in enumerate(models):
        explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional")
        sv = explainer.shap_values(X_eval, check_additivity=False)
        mean_abs = np.mean(np.abs(sv), axis=0)
        global_accum += mean_abs
        for f, val in zip(feature_cols, mean_abs):
            rows.append(
                {
                    "target_composition": target_cols[j],
                    "feature": f,
                    "mean_abs_shap": float(val),
                }
            )
    per_target = pd.DataFrame(rows).sort_values(["target_composition", "mean_abs_shap"], ascending=[True, False])
    global_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap_avg_over_targets": global_accum / len(models)})
    global_df = global_df.sort_values("mean_abs_shap_avg_over_targets", ascending=False).reset_index(drop=True)
    return per_target, global_df


def run_dataset(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    d_intrinsic: int,
    row_cap: int | None,
    output_dir: Path,
) -> None:
    valid = np.isfinite(df[feature_cols].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[valid].copy()
    if "event_id" in df.columns:
        df = df.sort_values("event_id")
    if row_cap is not None and len(df) > row_cap:
        df = df.iloc[:row_cap].copy()
    split = df["split"].astype(str).str.lower() if "split" in df.columns else pd.Series(["train"] * len(df))
    train_idx = (split == "train").to_numpy()
    test_idx = (split == "test").to_numpy()
    if test_idx.sum() == 0:
        test_idx = ~train_idx
    if train_idx.sum() < 50 or test_idx.sum() < 50:
        raise ValueError(f"{name}: insufficient train/test rows after filtering.")

    X_raw = df[feature_cols].to_numpy(dtype=float)
    sx = StandardScaler().fit(X_raw[train_idx])
    X = sx.transform(X_raw)

    # Composition target is in original feature basis but computed on standardized neighborhoods.
    C = local_subspace_composition(X, d=d_intrinsic, k=K_NEIGH)
    target_cols = [f"comp_{c}" for c in feature_cols]

    sy = StandardScaler().fit(C[train_idx])
    Y = sy.transform(C)

    models, Y_pred = fit_gbm_surrogates(X[train_idx], Y[train_idx], X[test_idx])
    r2_each = [float(r2_score(Y[test_idx, j], Y_pred[:, j])) for j in range(Y.shape[1])]

    # SHAP on test rows, background from train subset for speed/stability.
    rng = np.random.default_rng(42)
    tr_ids = np.where(train_idx)[0]
    bg_ids = rng.choice(tr_ids, size=min(500, len(tr_ids)), replace=False)
    te_ids = np.where(test_idx)[0]
    ev_ids = te_ids if len(te_ids) <= 1000 else rng.choice(te_ids, size=1000, replace=False)
    per_target_shap, global_shap = shap_importance(
        models=models,
        X_background=X[bg_ids],
        X_eval=X[ev_ids],
        feature_cols=feature_cols,
        target_cols=target_cols,
    )

    # Outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target_composition": target_cols,
            "test_r2": r2_each,
        }
    ).to_csv(output_dir / f"{name}_test_r2_by_target.csv", index=False)
    per_target_shap.to_csv(output_dir / f"{name}_shap_per_target.csv", index=False)
    global_shap.to_csv(output_dir / f"{name}_shap_global.csv", index=False)

    payload = {
        "dataset": name,
        "n_rows": int(len(df)),
        "n_train": int(train_idx.sum()),
        "n_test": int(test_idx.sum()),
        "k_neighbors": K_NEIGH,
        "intrinsic_dimension_used": int(d_intrinsic),
        "features": feature_cols,
        "mean_test_r2": float(np.mean(r2_each)),
        "median_test_r2": float(np.median(r2_each)),
        "top_global_features": global_shap.head(5).to_dict(orient="records"),
    }
    (output_dir / f"{name}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[{name}]")
    print(f"rows={len(df)} train={int(train_idx.sum())} test={int(test_idx.sum())} d={d_intrinsic} K={K_NEIGH}")
    print(f"mean test R2={np.mean(r2_each):.4f} | median test R2={np.median(r2_each):.4f}")
    print("top global SHAP features:")
    print(global_shap.head(8).to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5D latent MAP dataset
    p5 = PROJECT_ROOT / "Manifold Research" / "map_per_event_global.parquet"
    df5 = pd.read_parquet(p5, columns=["event_id", "split", "x", "e_y", "e_x", "psi", "omega_plus"])
    df5["psi"] = psi_to_radians(df5["psi"])
    run_dataset(
        name="latent5d",
        df=df5,
        feature_cols=["x", "e_y", "e_x", "psi", "omega_plus"],
        d_intrinsic=3,
        row_cap=6000,
        output_dir=OUT_DIR,
    )

    # 8D full physics+latent dataset from existing feature matrix
    p8 = PROJECT_ROOT / "outputs" / "capca_kernel_pca_collision_plus_pitch_6000" / "feature_matrix_used.csv"
    df8 = pd.read_csv(p8)
    run_dataset(
        name="full8d",
        df=df8,
        feature_cols=[
            "incoming_pitch_speed_mph",
            "incoming_pitch_spin_w_minus",
            "bat_speed_mph",
            "x",
            "psi",
            "e_y",
            "e_x",
            "omega_plus",
        ],
        d_intrinsic=6,
        row_cap=None,
        output_dir=OUT_DIR,
    )

    manifest = {
        "k_neighbors": K_NEIGH,
        "outputs_dir": str(OUT_DIR.resolve()),
        "datasets": ["latent5d", "full8d"],
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
