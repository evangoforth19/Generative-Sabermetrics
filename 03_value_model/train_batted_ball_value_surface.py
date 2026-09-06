#!/usr/bin/env python3
"""
Train a calibrated LightGBM multiclass surrogate: (EV, LA, SA) -> P(out, single, double, triple, home_run).

    python train_batted_ball_value_surface.py [--data-path PATH] [--output-dir DIR] [--no-calibration]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "batted_ball_value_surface"
DEFAULT_DATA = ROOT / "data" / "statcast_pybaseball" / "statcast_all.parquet"

CLASS_ORDER = ["out", "single", "double", "triple", "home_run"]

OUT_EVENTS = frozenset(
    {
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice_out",
        "sac_fly",
        "sac_fly_double_play",
        "triple_play",
    }
)

EXCLUDE_EVENTS = frozenset(
    {
        "field_error",
        "fielders_choice",
        "catcher_interf",
        "hit_by_pitch",
        "walk",
        "strikeout",
        "strikeout_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
    }
)

HOME_X = 125.42
HOME_Y = 198.27

FEATURE_COLS = ["launch_speed", "launch_angle", "sin_spray_angle", "cos_spray_angle"]


def discover_data_path(explicit: str | None) -> Path:
    if explicit is not None and str(explicit).strip():
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
    for p in (DEFAULT_DATA, ROOT / "data" / "statcast_pybaseball" / "statcast_all.parquet"):
        if p.is_file():
            return p.resolve()
    data_root = ROOT / "data"
    if data_root.is_dir():
        matches = [
            x
            for x in data_root.rglob("*")
            if x.is_file()
            and "statcast" in x.name.lower()
            and x.suffix.lower() in (".parquet", ".csv", ".feather")
        ]
        if matches:
            matches.sort(key=lambda x: x.stat().st_size, reverse=True)
            return matches[0].resolve()
    raise FileNotFoundError(
        "No local Statcast file found. Pass --data-path to a .parquet / .csv / .feather / .pkl file."
    )


def load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suf in (".feather", ".fea"):
        return pd.read_feather(path)
    if suf == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported format: {path}")


def spray_angle_deg(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    x = pd.to_numeric(hc_x, errors="coerce")
    y = pd.to_numeric(hc_y, errors="coerce")
    return np.degrees(np.arctan2(x - HOME_X, HOME_Y - y))


def bunt_mask(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    if "bb_type" in df.columns:
        m |= df["bb_type"].astype(str).str.lower().str.contains("bunt", na=False)
    for col in ("description", "des"):
        if col in df.columns:
            m |= df[col].astype(str).str.lower().str.contains("bunt", na=False)
    return m


def map_outcome_5class(events: pd.Series) -> pd.Series:
    e = events.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=events.index, dtype="string")
    out.loc[e == "single"] = "single"
    out.loc[e == "double"] = "double"
    out.loc[e == "triple"] = "triple"
    out.loc[e == "home_run"] = "home_run"
    out.loc[e.isin(OUT_EVENTS)] = "out"
    return out


def filter_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return filtered frame with outcome_5class and features; stats for metadata."""
    stats: dict[str, Any] = {"n_loaded": int(len(df))}
    need = {"launch_speed", "launch_angle", "events"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    if "spray_angle" in df.columns:
        sa = pd.to_numeric(df["spray_angle"], errors="coerce")
    elif {"hc_x", "hc_y"}.issubset(df.columns):
        sa = spray_angle_deg(df["hc_x"], df["hc_y"])
    else:
        raise ValueError("Need spray_angle or (hc_x, hc_y) for spray angle.")

    ev = pd.to_numeric(df["launch_speed"], errors="coerce")
    la = pd.to_numeric(df["launch_angle"], errors="coerce")
    ok = ev.notna() & la.notna() & sa.notna()
    d = df.loc[ok].copy()
    d["_ev"] = ev.loc[ok]
    d["_la"] = la.loc[ok]
    d["_sa"] = sa.loc[ok]

    evts = d["events"].astype(str).str.strip().str.lower()
    d = d.loc[~evts.isin(EXCLUDE_EVENTS)].copy()
    d = d.loc[~bunt_mask(d)].copy()

    d["outcome_5class"] = map_outcome_5class(d["events"])
    d = d.loc[d["outcome_5class"].notna()].copy()

    rad = np.radians(d["_sa"].to_numpy(dtype=np.float64))
    d["launch_speed"] = d["_ev"].astype(np.float64)
    d["launch_angle"] = d["_la"].astype(np.float64)
    d["sin_spray_angle"] = np.sin(rad)
    d["cos_spray_angle"] = np.cos(rad)

    stats["n_after_filter"] = int(len(d))
    return d, stats


def assign_year(df: pd.DataFrame) -> pd.Series:
    if "game_date" in df.columns:
        y = pd.to_datetime(df["game_date"], errors="coerce").dt.year
    else:
        y = pd.Series(pd.NA, index=df.index)
    if "game_year" in df.columns:
        gy = pd.to_numeric(df["game_year"], errors="coerce")
        y = y.fillna(gy)
    return y.astype("Int64")


def split_data(
    df: pd.DataFrame,
    *,
    val_year: int,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ycol = assign_year(df)
    df = df.assign(_split_year=ycol)
    years = set(df["_split_year"].dropna().astype(int).tolist())
    use_chrono = val_year in years and test_year in years and any(y < val_year for y in years)
    if use_chrono:
        tr = df.loc[df["_split_year"] < val_year].copy()
        va = df.loc[df["_split_year"] == val_year].copy()
        te = df.loc[df["_split_year"] == test_year].copy()
        if len(tr) < 500 or len(va) < 100 or len(te) < 50:
            use_chrono = False
    if use_chrono:
        method = f"chronological train year<{val_year}, val={val_year}, test={test_year}"
        return tr, va, te, method

    y_str = df["outcome_5class"].astype(str)
    idx = np.arange(len(df))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=y_str)
    y_tmp = y_str.iloc[tmp_idx].reset_index(drop=True)
    va_rel, te_rel = train_test_split(
        np.arange(len(tmp_idx)), test_size=0.5, random_state=43, stratify=y_tmp
    )
    va_idx = tmp_idx[va_rel]
    te_idx = tmp_idx[te_rel]
    tr = df.iloc[tr_idx].copy()
    va = df.iloc[va_idx].copy()
    te = df.iloc[te_idx].copy()
    method = "stratified 70% / 15% / 15% (fallback: chronological split not viable)"
    return tr, va, te, method


def multiclass_nll(y_true: np.ndarray, P: np.ndarray, eps: float = 1e-15) -> float:
    """y_true integer labels 0..K-1; P row-normalized probs."""
    n = len(y_true)
    p = P[np.arange(n), y_true.astype(int)]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def multiclass_brier(y_true: np.ndarray, P: np.ndarray) -> float:
    n, k = P.shape
    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def classwise_brier(y_true: np.ndarray, P: np.ndarray, n_classes: int) -> dict[str, float]:
    out = {}
    for k in range(n_classes):
        yk = (y_true == k).astype(np.float64)
        pk = P[:, k]
        out[CLASS_ORDER[k]] = float(np.mean((pk - yk) ** 2))
    return out


def ece_top1(y_true: np.ndarray, P: np.ndarray, n_bins: int = 15) -> float:
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        w = m.sum()
        if w == 0:
            continue
        ece += (w / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def ece_ovr(y_true: np.ndarray, P: np.ndarray, k: int, n_bins: int = 15) -> float:
    pk = P[:, k]
    yk = (y_true == k).astype(np.float64)
    order = np.argsort(pk)
    pk_s, yk_s = pk[order], yk[order]
    n = len(pk_s)
    ece = 0.0
    for b in range(n_bins):
        lo = int(b * n / n_bins)
        hi = int((b + 1) * n / n_bins)
        if lo >= hi:
            continue
        w = hi - lo
        ece += (w / n) * abs(yk_s[lo:hi].mean() - pk_s[lo:hi].mean())
    return float(ece)


def softmax_rows(Z: np.ndarray) -> np.ndarray:
    z = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def fit_temperature(raw_val: np.ndarray, y_val: np.ndarray) -> float:
    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        P = softmax_rows(raw_val / T)
        return multiclass_nll(y_val, P)

    res = minimize_scalar(nll, bounds=(0.05, 50.0), method="bounded")
    return float(res.x)


def train_main(args: argparse.Namespace) -> None:
    data_path = discover_data_path(args.data_path)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUT.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    template_model = ROOT / "outputs" / "batted_ball_value_surface" / "batted_ball_value_model.py"
    dst_model = out_dir / "batted_ball_value_model.py"
    if template_model.is_file() and template_model.resolve() != dst_model.resolve():
        shutil.copy2(template_model, dst_model)

    df = load_table(data_path)
    df_f, load_stats = filter_dataset(df)

    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    idx_to_class = dict(enumerate(CLASS_ORDER))

    tr, va, te, split_method = split_data(
        df_f.assign(outcome_5class=df_f["outcome_5class"]),
        val_year=args.val_year,
        test_year=args.test_year,
    )

    def Xpart(sub: pd.DataFrame) -> np.ndarray:
        return sub[FEATURE_COLS].to_numpy(dtype=np.float64)

    y_tr = tr["outcome_5class"].astype(str).map(class_to_idx).to_numpy(np.int32)
    y_va = va["outcome_5class"].astype(str).map(class_to_idx).to_numpy(np.int32)
    y_te = te["outcome_5class"].astype(str).map(class_to_idx).to_numpy(np.int32)

    from lightgbm import LGBMClassifier

    try:
        from lightgbm import early_stopping

        _cb = [early_stopping(stopping_rounds=50, verbose=False)]
        use_cb = True
    except ImportError:
        _cb = None
        use_cb = False

    clf = LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    fit_kw: dict[str, Any] = {
        "X": Xpart(tr),
        "y": y_tr,
        "eval_set": [(Xpart(va), y_va)],
        "eval_metric": "multi_logloss",
    }
    if use_cb:
        try:
            clf.fit(**fit_kw, callbacks=_cb)
        except TypeError:
            clf.fit(**fit_kw, early_stopping_rounds=50)
    else:
        clf.fit(**fit_kw, early_stopping_rounds=50)

    raw_va = clf.predict(Xpart(va), raw_score=True)
    raw_te = clf.predict(Xpart(te), raw_score=True)
    P_va_unc = softmax_rows(raw_va)
    P_te_unc = softmax_rows(raw_te)

    nll_va_before = multiclass_nll(y_va, P_va_unc)
    nll_te_before = multiclass_nll(y_te, P_te_unc)
    brier_va_before = multiclass_brier(y_va, P_va_unc)
    brier_te_before = multiclass_brier(y_te, P_te_unc)
    ece_va_before = ece_top1(y_va, P_va_unc)
    ece_te_before = ece_top1(y_te, P_te_unc)

    if args.no_calibration:
        T = 1.0
        P_va_cal = P_va_unc
        P_te_cal = P_te_unc
    else:
        T = fit_temperature(raw_va, y_va)
        P_va_cal = softmax_rows(raw_va / T)
        P_te_cal = softmax_rows(raw_te / T)

    nll_va_after = multiclass_nll(y_va, P_va_cal)
    nll_te_after = multiclass_nll(y_te, P_te_cal)
    brier_va_after = multiclass_brier(y_va, P_va_cal)
    brier_te_after = multiclass_brier(y_te, P_te_cal)
    ece_va_after = ece_top1(y_va, P_va_cal)
    ece_te_after = ece_top1(y_te, P_te_cal)

    cw_b_va = classwise_brier(y_va, P_va_cal, 5)
    cw_b_te = classwise_brier(y_te, P_te_cal, 5)
    ovr_ece_te = {CLASS_ORDER[k]: ece_ovr(y_te, P_te_cal, k) for k in range(5)}
    reliability_te = {
        CLASS_ORDER[k]: {
            "mean_predicted_probability": float(P_te_cal[:, k].mean()),
            "empirical_frequency": float((y_te == k).mean()),
        }
        for k in range(5)
    }

    triple_train = int((tr["outcome_5class"] == "triple").sum())
    triple_warn = triple_train < 500

    meta = {
        "data_path": str(data_path),
        "n_rows_loaded": load_stats["n_loaded"],
        "n_rows_after_filter": load_stats["n_after_filter"],
        "feature_columns": FEATURE_COLS,
        "class_order": CLASS_ORDER,
        "split_method": split_method,
        "train_size": len(tr),
        "val_size": len(va),
        "test_size": len(te),
        "hyperparameters": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
        },
        "class_weighting": "balanced (LGBMClassifier)",
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "calibration_temperature.json").write_text(
        json.dumps({"temperature": T, "calibration": "temperature_scaled_softmax_raw_scores"}, indent=2),
        encoding="utf-8",
    )
    with open(out_dir / "model_lgbm.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f)

    bundle = {"classifier": clf, "temperature": T, "metadata": meta}

    mod_path = out_dir / "batted_ball_value_model.py"
    spec = importlib.util.spec_from_file_location("bbvm", mod_path)
    assert spec and spec.loader
    bbvm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bbvm)

    examples_rows = []
    for ev, la, sa in [(105, 25, 0), (80, -10, 10), (98, 12, -25), (110, 35, 35)]:
        p = bbvm.predict_outcome_probs(ev, la, sa, bundle)
        examples_rows.append(
            {
                "EV": ev,
                "LA": la,
                "SA": sa,
                "P(out)": float(p[0]),
                "P(single)": float(p[1]),
                "P(double)": float(p[2]),
                "P(triple)": float(p[3]),
                "P(home_run)": float(p[4]),
                "xwOBAcon_3D": float(bbvm.predict_xwobacon3d(ev, la, sa, bundle)),
            }
        )

    def pct_counts(sub: pd.DataFrame) -> dict[str, Any]:
        vc = sub["outcome_5class"].value_counts()
        tot = len(sub)
        return {str(k): {"count": int(v), "pct": float(100 * v / tot)} for k, v in vc.items()}

    md_lines = [
        "# Batted-ball outcome surrogate — calibration summary",
        "",
        "## Data",
        f"- **Dataset path:** `{data_path}`",
        f"- **Rows loaded:** {load_stats['n_loaded']:,}",
        f"- **Rows after EV/LA/SA + outcome filters (no bunts / excluded events):** {load_stats['n_after_filter']:,}",
        "",
        "## Model",
        f"- **Features (only):** `{FEATURE_COLS}`",
        f"- **Target classes (order):** `{CLASS_ORDER}`",
        f"- **Split:** {split_method}",
        f"- **Train / validation / test sizes:** {len(tr):,} / {len(va):,} / {len(te):,}",
        "",
        "### Class distribution (counts and %)",
        "**Train:**",
        "```json",
        json.dumps(pct_counts(tr), indent=2),
        "```",
        "**Validation:**",
        "```json",
        json.dumps(pct_counts(va), indent=2),
        "```",
        "**Test:**",
        "```json",
        json.dumps(pct_counts(te), indent=2),
        "```",
        "",
        "### LightGBM hyperparameters",
        "```json",
        json.dumps(meta["hyperparameters"], indent=2),
        "```",
        f"- **Class weighting:** {meta['class_weighting']}",
        "",
        "## Calibration metrics",
        "",
        "| Metric | Validation (before) | Validation (after) | Test (before) | Test (after) |",
        "|--------|----------------------|--------------------|---------------|--------------|",
        f"| Multiclass NLL | {nll_va_before:.5f} | {nll_va_after:.5f} | {nll_te_before:.5f} | {nll_te_after:.5f} |",
        f"| Multiclass Brier | {brier_va_before:.5f} | {brier_va_after:.5f} | {brier_te_before:.5f} | {brier_te_after:.5f} |",
        f"| ECE (top-1, 15 bins) | {ece_va_before:.5f} | {ece_va_after:.5f} | {ece_te_before:.5f} | {ece_te_after:.5f} |",
        "",
        f"- **Fitted temperature T:** {T:.6f}",
        "",
        "### Class-wise Brier (test, after calibration)",
        "```json",
        json.dumps(cw_b_te, indent=2),
        "```",
        "",
        "### Class-wise one-vs-rest ECE (test, after calibration, 15 bins)",
        "```json",
        json.dumps(ovr_ece_te, indent=2),
        "```",
        "",
        "### Class-wise reliability (test, after calibration)",
        "Mean predicted probability vs empirical frequency of each class.",
        "```json",
        json.dumps(reliability_te, indent=2),
        "```",
        "",
        "## Plain-English interpretation",
        "Lower NLL and Brier on validation/test after temperature scaling indicate better probability quality for simulation use. "
        "ECE summarizes how much average predicted confidence deviates from realized hit frequencies (smaller is better). "
        "Triples are extremely sparse; treat their predicted probabilities as noisy unless counts are large.",
        "",
    ]
    if triple_warn:
        md_lines.append(
            f"> **Warning:** Training set triple count is only **{triple_train}** (< 500). "
            "Triple calibration and probabilities may be unstable for Monte Carlo use."
        )
        md_lines.append("")
    md_lines += [
        "## Sanity-check predictions (calibrated)",
        "",
        "| EV | LA | SA | P(out) | P(single) | P(double) | P(triple) | P(home_run) | xwOBAcon_3D |",
        "|----|----|----|--------|-----------|-----------|-----------|-------------|--------------|",
    ]
    for row in examples_rows:
        md_lines.append(
            f"| {row['EV']} | {row['LA']} | {row['SA']} | {row['P(out)']:.4f} | {row['P(single)']:.4f} | "
            f"{row['P(double)']:.4f} | {row['P(triple)']:.4f} | {row['P(home_run)']:.4f} | {row['xwOBAcon_3D']:.4f} |"
        )
    md_lines.append("")

    summary_path = out_dir / "calibration_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    summ = out_dir / "calibration_summary.md"
    print(str(out_dir / "model_lgbm.pkl"))
    print(str(summ))
    print(f"{nll_te_after:.6f}")
    print(f"{brier_te_after:.6f}")
    print(f"{ece_te_after:.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--val-year", type=int, default=2025)
    p.add_argument("--test-year", type=int, default=2026)
    p.add_argument("--no-calibration", action="store_true")
    args = p.parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
