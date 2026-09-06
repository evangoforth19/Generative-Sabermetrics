#!/usr/bin/env python3
"""
Test whether MAP collision latents lie near a 2D surface parameterized by (x, psi).

Hypothesis (H):
  On the support of the MAP cloud, (e_y, e_x, omega_plus) ≈ F(x, psi) with F nonlinear;
  x and psi are preferred intrinsic coordinates. Cross-event variation collapses to two
  directions; this is distinct from per-event algebraic constraint dimension.

Procedure:
  - Standardize all five latents using train split statistics only.
  - Forward: for each candidate feature pair (two latents), fit HistGradientBoosting
    regressors (train) predicting the other three latents; report R2 / RMSE on val and test.
  - Inverse: from (e_y, e_x, omega_plus) predict x and psi separately (asymmetry check).
  - Residual checks vs log_target (Spearman) on test after (x, psi) forward fit.
  - Partial dependence plots for (x, psi) -> each target (train fit).

Run from project root:
    python scripts/run_map_latent_x_psi_hypothesis.py
    python scripts/run_map_latent_x_psi_hypothesis.py --input Manifold\\ Research/map_per_event_global.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PREFERRED = [
    PROJECT_ROOT / "Manifold Research" / "map_per_event_global.parquet",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "map_latent_x_psi_hypothesis"

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

LATENT_ORDER = ["x", "e_y", "e_x", "psi", "omega_plus"]

HYPOTHESIS_TEXT = (
    "H: On the support of the MAP / posterior cloud, the five collision latents lie near "
    "a 2D surface parameterized by (x, psi): (e_y, e_x, omega_plus) ≈ F(x, psi), with F "
    "possibly nonlinear; x and psi are the proposed minimal coordinates. This is a "
    "global across-events claim, distinct from per-event algebraic constraint dimension."
)


def discover_input(root: Path) -> Path | None:
    for p in DEFAULT_INPUT_PREFERRED:
        if p.exists():
            return p
    for pat in ("**/map_per_event_global.parquet", "**/event_level_targets.parquet"):
        found = sorted(
            (q for q in root.glob(pat) if q.is_file() and q.suffix.lower() == ".parquet"),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        )
        if found:
            return found[0]
    return None


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


def load_latent_frame(
    input_path: Path,
) -> tuple[pd.DataFrame, dict[str, str], pd.Series | None, pd.Series | None]:
    df = pd.read_parquet(input_path) if input_path.suffix.lower() == ".parquet" else pd.read_csv(input_path)
    resolved = resolve_columns(df)
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
    split_col = None
    if "split" in df.columns:
        split_col = df.loc[valid, "split"].astype(str).reset_index(drop=True)
    log_target = None
    if "log_target" in df.columns:
        log_target = pd.to_numeric(df.loc[valid, "log_target"], errors="coerce").reset_index(drop=True)
    return feat, resolved, split_col, log_target


def default_hgbr() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=300,
        learning_rate=0.06,
        l2_regularization=1e-2,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=42,
    )


def fit_predict_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    model = default_hgbr()
    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_eval)
    r2 = float(r2_score(y_eval, pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, pred)))
    return pred, r2, rmse


def forward_block(
    Z: pd.DataFrame,
    pair: list[str],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> dict[str, Any]:
    others = [c for c in LATENT_ORDER if c not in pair]
    X_tr = Z.loc[train_idx, pair].to_numpy(dtype=float)
    X_ev = Z.loc[eval_idx, pair].to_numpy(dtype=float)
    preds = []
    rows = []
    y_true_mat = Z.loc[eval_idx, others].to_numpy(dtype=float)
    y_pred_mat = np.zeros_like(y_true_mat)
    for j, t in enumerate(others):
        y_tr = Z.loc[train_idx, t].to_numpy(dtype=float)
        y_ev = Z.loc[eval_idx, t].to_numpy(dtype=float)
        p, r2, rmse = fit_predict_one(X_tr, y_tr, X_ev, y_ev)
        y_pred_mat[:, j] = p
        preds.append(p)
        rows.append({"target": t, "r2": r2, "rmse": rmse})
    joint_rmse = float(np.sqrt(mean_squared_error(y_true_mat.ravel(), y_pred_mat.ravel())))
    mean_r2 = float(np.mean([r["r2"] for r in rows]))
    return {"per_target": rows, "mean_r2": mean_r2, "joint_rmse_all_three": joint_rmse, "y_true": y_true_mat, "y_pred": y_pred_mat}


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    r = pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman")
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test (x, psi) as 2D coordinates for MAP collision latents.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = discover_input(PROJECT_ROOT)
    if input_path is None or not input_path.exists():
        print("No input file found. Pass --input.", file=sys.stderr)
        sys.exit(1)

    feat, resolved, split_col, log_target = load_latent_frame(input_path)
    if split_col is None:
        print("Column 'split' not found; need train/val/test for leakage-safe metrics.", file=sys.stderr)
        sys.exit(1)

    split_norm = split_col.str.lower().str.strip()
    train_idx = np.where(split_norm == "train")[0]
    val_idx = np.where(split_norm == "val")[0]
    test_idx = np.where(split_norm == "test")[0]
    if train_idx.size < 100 or test_idx.size < 50:
        print("Insufficient rows in train or test split.", file=sys.stderr)
        sys.exit(1)

    scaler = StandardScaler()
    scaler.fit(feat.iloc[train_idx])
    Z = pd.DataFrame(scaler.transform(feat), columns=LATENT_ORDER)

    pairs: list[tuple[str, tuple[str, str]]] = [
        ("x_psi", ("x", "psi")),
        ("x_e_x", ("x", "e_x")),
        ("x_e_y", ("x", "e_y")),
        ("x_omega_plus", ("x", "omega_plus")),
        ("psi_e_x", ("psi", "e_x")),
        ("psi_e_y", ("psi", "e_y")),
        ("psi_omega_plus", ("psi", "omega_plus")),
        ("e_y_e_x", ("e_y", "e_x")),
        ("e_y_omega_plus", ("e_y", "omega_plus")),
        ("e_x_omega_plus", ("e_x", "omega_plus")),
    ]

    rows_forward: list[dict[str, Any]] = []
    for name, pair in pairs:
        for split_name, idx in ("val", val_idx), ("test", test_idx):
            out = forward_block(Z, list(pair), train_idx, idx)
            for r in out["per_target"]:
                rows_forward.append(
                    {
                        "pair": name,
                        "feature_a": pair[0],
                        "feature_b": pair[1],
                        "eval_split": split_name,
                        "target": r["target"],
                        "r2": r["r2"],
                        "rmse": r["rmse"],
                        "mean_r2_three_targets": out["mean_r2"],
                        "joint_rmse_three_targets": out["joint_rmse_all_three"],
                    }
                )

    # Inverse: (e_y, e_x, omega_plus) -> x and psi
    inv_feats = ["e_y", "e_x", "omega_plus"]
    X_tr_inv = Z.loc[train_idx, inv_feats].to_numpy(float)
    rows_inverse: list[dict[str, Any]] = []
    for split_name, idx in ("val", val_idx), ("test", test_idx):
        X_ev = Z.loc[idx, inv_feats].to_numpy(float)
        for target in ("x", "psi"):
            _, r2, rmse = fit_predict_one(
                X_tr_inv,
                Z.loc[train_idx, target].to_numpy(float),
                X_ev,
                Z.loc[idx, target].to_numpy(float),
            )
            rows_inverse.append(
                {
                    "direction": "inverse_e_y_e_x_omega_to_latent",
                    "eval_split": split_name,
                    "target": target,
                    "r2": r2,
                    "rmse": rmse,
                }
            )

    # Primary forward on test for residuals vs log_target
    residual_rows: list[dict[str, Any]] = []
    if log_target is not None:
        out_test = forward_block(Z, ["x", "psi"], train_idx, test_idx)
        lt = log_target.iloc[test_idx].to_numpy(dtype=float)
        others = [c for c in LATENT_ORDER if c not in ("x", "psi")]
        for j, t in enumerate(others):
            res = out_test["y_true"][:, j] - out_test["y_pred"][:, j]
            rho = spearman_safe(lt, res)
            rho_abs = spearman_safe(lt, np.abs(res))
            residual_rows.append(
                {
                    "forward_pair": "x_psi",
                    "eval_split": "test",
                    "target": t,
                    "spearman_log_target_residual": rho,
                    "spearman_log_target_abs_residual": rho_abs,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows_forward).to_csv(args.output_dir / "forward_metrics_by_pair.csv", index=False)
    pd.DataFrame(rows_inverse).to_csv(args.output_dir / "inverse_metrics_e_y_e_x_omega.csv", index=False)
    if residual_rows:
        pd.DataFrame(residual_rows).to_csv(args.output_dir / "residual_vs_log_target_test.csv", index=False)

    # Summary: best mean R2 on test per pair (mean across the three predicted targets — one row per pair per split)
    df_f = pd.DataFrame(rows_forward)
    summ = (
        df_f.groupby(["pair", "eval_split"], as_index=False)
        .agg(mean_r2_three_targets=("mean_r2_three_targets", "first"), joint_rmse=("joint_rmse_three_targets", "first"))
        .sort_values(["eval_split", "mean_r2_three_targets"], ascending=[True, False])
    )
    summ.to_csv(args.output_dir / "forward_summary_mean_r2_by_pair.csv", index=False)

    # PDP for (x, psi) models on train
    primary = ["x", "psi"]
    others = [c for c in LATENT_ORDER if c not in primary]
    fig, axes = plt.subplots(len(others), 2, figsize=(8, 3 * len(others)), squeeze=False)
    for i, t in enumerate(others):
        model = default_hgbr()
        model.fit(
            Z.loc[train_idx, primary].to_numpy(float),
            Z.loc[train_idx, t].to_numpy(float),
        )
        for j, feat_name in enumerate(primary):
            PartialDependenceDisplay.from_estimator(
                model,
                Z.loc[train_idx, primary],
                features=[j],
                ax=axes[i, j],
                feature_names=primary,
            )
            axes[i, j].set_title(f"PDP {feat_name} -> {t} (train)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "partial_dependence_x_psi_train.png", dpi=150)
    plt.close()

    manifest = {
        "hypothesis": HYPOTHESIS_TEXT,
        "input_path": str(input_path.resolve()),
        "resolved_columns": resolved,
        "n_rows_total": int(len(feat)),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "model": "HistGradientBoostingRegressor (see script default_hgbr)",
        "latent_order": LATENT_ORDER,
        "note_conditional_independence": (
            "Residual vs log_target Spearman on test is a weak check for leftover structure. "
            "pitch_type / park / batter require merging other tables if desired."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(HYPOTHESIS_TEXT)
    print("\nWrote:", args.output_dir)
    print("\nForward summary (test, sorted by mean R2 of three targets):\n")
    print(summ[summ["eval_split"] == "test"].to_string(index=False))
    print("\nInverse (e_y, e_x, omega_plus) -> x, psi on test:\n")
    print(pd.DataFrame(rows_inverse)[pd.DataFrame(rows_inverse)["eval_split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
