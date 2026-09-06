from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def main() -> None:
    root = Path("/home/evangoforth03/Bayesian Research")
    feature_path = root / "outputs/local_pca_map_mstar_exit_inputs_6000/local_pca_feature_matrix_used.csv"
    map_path = root / "Manifold Research/map_per_event_global.parquet"
    out_dir = root / "outputs/e_y_regression_collision_vars_map6000_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(feature_path)
    map_df = pd.read_parquet(map_path)[["event_id", "x", "psi", "e_y"]].drop_duplicates("event_id")

    df = features.merge(map_df, on="event_id", how="left")

    df["pitch_speed_mag_1d"] = np.sqrt(df["vx0"] ** 2 + df["vy0"] ** 2 + df["vz0"] ** 2)
    df["bat_speed_1d"] = df["bat_speed_obs_mph"]
    df["L_minus_x"] = df["L_in"] - df["x"]

    predictors = ["pitch_speed_mag_1d", "bat_speed_1d", "L_minus_x", "x", "psi"]
    target = "e_y"
    needed = ["event_id", "split"] + predictors + [target]
    df = df[needed].dropna().copy()

    train_mask = df["split"].astype(str).str.lower().eq("train")
    test_mask = df["split"].astype(str).str.lower().eq("test")

    X = df[predictors].to_numpy()
    y = df[target].to_numpy()
    X_train = df.loc[train_mask, predictors].to_numpy()
    y_train = df.loc[train_mask, target].to_numpy()
    X_test = df.loc[test_mask, predictors].to_numpy()
    y_test = df.loc[test_mask, target].to_numpy()

    metrics_rows: list[dict[str, float | int | str]] = []
    for degree, name in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("lin", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)

        yhat_all = model.predict(X)
        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        n_features = int(model.named_steps["poly"].n_output_features_)
        metrics_rows.append(
            {
                "model": name,
                "degree": degree,
                "n_features": n_features,
                "r2_all": float(r2_score(y, yhat_all)),
                "rmse_all": float(np.sqrt(mean_squared_error(y, yhat_all))),
                "r2_train": float(r2_score(y_train, yhat_train)),
                "rmse_train": float(np.sqrt(mean_squared_error(y_train, yhat_train))),
                "r2_test": float(r2_score(y_test, yhat_test)),
                "rmse_test": float(np.sqrt(mean_squared_error(y_test, yhat_test))),
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "regression_metrics.csv", index=False)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gamma = 1.0 / Xs.shape[1]
    kpca = KernelPCA(kernel="rbf", gamma=gamma, n_components=min(Xs.shape[0], 200))
    kpca.fit(Xs)

    lambdas = np.asarray(kpca.eigenvalues_, dtype=float)
    lambdas = np.maximum(lambdas, 0.0)
    total = float(lambdas.sum())
    var_ratio = lambdas / total if total > 0 else np.zeros_like(lambdas)
    cum_ratio = np.cumsum(var_ratio)
    top_n = min(10, len(lambdas))
    top10 = pd.DataFrame(
        {
            "component": np.arange(1, top_n + 1),
            "eigenvalue": lambdas[:top_n],
            "explained_variance_ratio": var_ratio[:top_n],
            "cumulative_explained_variance_ratio": cum_ratio[:top_n],
        }
    )
    top10.to_csv(out_dir / "rbf_kernel_pca_top10.csv", index=False)

    df.to_csv(out_dir / "analysis_dataset_used.csv", index=False)
    summary = {
        "n_samples_used": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "rbf_gamma": gamma,
        "predictors": predictors,
        "target": target,
        "source_target": str(map_path),
        "source_predictors": str(feature_path),
        "notes": "Uses production 6000 events joined to per-event MAP x/psi/e_y; excludes e_y from predictors.",
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
