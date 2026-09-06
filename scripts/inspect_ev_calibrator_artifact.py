#!/usr/bin/env python3
"""
Inspect EV calibrator artifact deeply and test prediction reconstruction.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def soft_cap(y: np.ndarray, cap: float = 121.0, tau: float = 2.0) -> np.ndarray:
    x = (cap - y) / tau
    return cap - tau * np.logaddexp(0.0, x)


def fmt(x: Any, nd: int = 6) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    av = a[m]
    bv = b[m]
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def summarize_obj(obj: Any) -> str:
    t = type(obj)
    if isinstance(obj, pd.DataFrame):
        return f"type={t}, shape={obj.shape}, cols={list(obj.columns)[:12]}"
    if isinstance(obj, pd.Series):
        return f"type={t}, shape={obj.shape}, name={obj.name}"
    if isinstance(obj, np.ndarray):
        return f"type={t}, shape={obj.shape}, dtype={obj.dtype}"
    if isinstance(obj, dict):
        return f"type={t}, len={len(obj)}, keys={list(obj.keys())[:12]}"
    if isinstance(obj, (list, tuple, set)):
        return f"type={t}, len={len(obj)}"
    return f"type={t}, repr={repr(obj)[:240]}"


def walk_tree(
    obj: Any,
    lines: list[str],
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
) -> None:
    if depth > max_depth:
        return
    lines.append(f"{prefix}{summarize_obj(obj)}")
    if depth == max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(f"{prefix}  key={k!r}")
            walk_tree(v, lines, prefix + "    ", depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:20]):
            lines.append(f"{prefix}  idx={i}")
            walk_tree(v, lines, prefix + "    ", depth + 1, max_depth)


def find_model_like(obj: Any, found: list[tuple[str, Any]], path: str = "root", depth: int = 0, max_depth: int = 6) -> None:
    if depth > max_depth:
        return
    attrs = set(dir(obj))
    modelish = False
    if "fit" in attrs and ("predict" in attrs or "transform" in attrs):
        modelish = True
    if "named_steps" in attrs or "get_params" in attrs:
        modelish = True
    if modelish:
        found.append((path, obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_model_like(v, found, f"{path}.{k}", depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_model_like(v, found, f"{path}[{i}]", depth + 1, max_depth)


def infer_cols(df: pd.DataFrame, is_cal: bool) -> dict[str, str | None]:
    out = df.copy()
    eid = "event_id" if "event_id" in out.columns else None
    if eid is None and all(c in out.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
        eid = "__event_id__"
        out[eid] = (
            out["game_pk"].astype(str)
            + "_"
            + out["at_bat_number"].astype(str)
            + "_"
            + out["pitch_number"].astype(str)
        )
    if eid is None:
        raise ValueError("Could not infer event id column")
    dec = next((c for c in ["EV_dec", "ev_dec", "EV", "launch_speed_dec", "decoded_EV"] if c in out.columns), None)
    obs = next((c for c in ["EV_obs", "ev_obs", "observed_EV", "launch_speed"] if c in out.columns), None)
    cal = next((c for c in ["EV_cal", "ev_cal", "calibrated_EV"] if c in out.columns), None) if is_cal else None
    if dec is None:
        raise ValueError("Could not infer EV_dec")
    if obs is None:
        raise ValueError("Could not infer EV_obs")
    if is_cal and cal is None:
        raise ValueError("Could not infer EV_cal")
    return {"event_id": eid, "ev_dec": dec, "ev_obs": obs, "ev_cal": cal}


def build_event_summary(df: pd.DataFrame, c: dict[str, str | None], use_cal: bool) -> pd.DataFrame:
    eid = c["event_id"]
    ev = c["ev_cal"] if use_cal else c["ev_dec"]
    assert eid is not None and ev is not None
    d = df.copy()
    d[eid] = d[eid].astype(str)
    d[ev] = pd.to_numeric(d[ev], errors="coerce")
    d[c["ev_dec"]] = pd.to_numeric(d[c["ev_dec"]], errors="coerce")
    d[c["ev_obs"]] = pd.to_numeric(d[c["ev_obs"]], errors="coerce")
    g = d.groupby(eid, sort=False)
    out = pd.DataFrame(
        {
            "event_id": g.size().index.astype(str),
            "EV_obs": g[c["ev_obs"]].first().to_numpy(),
            "EV_dec_mean": g[c["ev_dec"]].mean().to_numpy(),
            "EV_target_mean": g[ev].mean().to_numpy(),
            "n_draws": g.size().to_numpy(),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrator-path", type=Path, required=True)
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument(
        "--uncalibrated-draws-path",
        type=Path,
        default=Path("artifacts/physics_decoder_calibration_diagnostics/20260508_0319Z/uncal_with_obs.parquet"),
    )
    ap.add_argument(
        "--calibrated-draws-path",
        type=Path,
        default=Path("artifacts/physics_decoder_calibration_diagnostics/20260508_0319Z/cal_with_obs.parquet"),
    )
    args = ap.parse_args()

    lines: list[str] = []
    cp = args.calibrator_path.resolve()
    lines.append("EV CALIBRATOR ARTIFACT INSPECTION")
    lines.append("=" * 80)
    lines.append(f"calibrator_path: {cp}")
    lines.append(f"exists: {cp.exists()}")
    if not cp.exists():
        raise FileNotFoundError(cp)

    obj = joblib.load(cp)
    lines.append("")
    lines.append("[1] TOP-LEVEL OBJECT INFO")
    lines.append("-" * 80)
    lines.append(f"python_type: {type(obj)}")
    lines.append(f"repr_summary: {repr(obj)[:1200]}")
    if isinstance(obj, dict):
        lines.append(f"top_level_keys: {list(obj.keys())}")
    else:
        lines.append("top_level_keys: <not a dict>")

    lines.append("")
    lines.append("nested_key_tree (depth<=4):")
    walk_lines: list[str] = []
    walk_tree(obj, walk_lines, max_depth=4)
    lines.extend(walk_lines[:3000])

    lines.append("")
    lines.append("[2] MODEL OBJECT INSPECTION")
    lines.append("-" * 80)
    found: list[tuple[str, Any]] = []
    find_model_like(obj, found)
    # dedupe by id
    seen: set[int] = set()
    uniq = []
    for p, m in found:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq.append((p, m))
    if not uniq:
        lines.append("No model-like objects found.")
    for path, model in uniq:
        lines.append(f"path={path}")
        lines.append(f"  type={type(model)}")
        if hasattr(model, "get_params"):
            try:
                gp = model.get_params(deep=False)
                lines.append(f"  get_params(keys)={list(gp.keys())[:80]}")
                lines.append(f"  get_params(summary)={repr(gp)[:1600]}")
            except Exception as e:
                lines.append(f"  get_params failed: {e}")
        if hasattr(model, "feature_names_in_"):
            try:
                lines.append(f"  feature_names_in_={list(model.feature_names_in_)}")
            except Exception:
                pass
        if hasattr(model, "named_steps"):
            try:
                ns = model.named_steps
                lines.append(f"  named_steps={list(ns.keys())}")
                if len(ns):
                    last = list(ns.values())[-1]
                    lines.append(f"  final_estimator_class={type(last)}")
            except Exception as e:
                lines.append(f"  named_steps inspect failed: {e}")

    lines.append("")
    lines.append("[3] FEATURE METADATA")
    lines.append("-" * 80)
    wanted = [
        "feature_cols",
        "numeric_cols",
        "categorical_cols",
        "target_col",
        "residual_col",
        "ev_dec_col",
        "ev_obs_col",
        "ev_cal_col",
        "rho",
        "cap",
        "tau",
        "training_config",
        "metrics",
        "manifest",
        "numeric_features",
        "categorical_features",
        "target_name",
        "ev_col",
        "group_col",
    ]
    if isinstance(obj, dict):
        for k in wanted:
            if k in obj:
                lines.append(f"{k}: {repr(obj[k])[:2500]}")
    # If payload has custom keys + pipeline
    if isinstance(obj, dict):
        for k in obj.keys():
            if any(tok in str(k).lower() for tok in ("feature", "target", "resid", "rho", "cap", "tau", "ev_")):
                lines.append(f"key_match[{k}]: {repr(obj[k])[:600]}")

    lines.append("")
    lines.append("[4] LEAKAGE / TARGET AUDIT")
    lines.append("-" * 80)
    tokens = [
        "obs",
        "observed",
        "launch_speed",
        "ev_obs",
        "residual",
        "target",
        "ev_cal",
        "estimated_ba_using_speedangle",
        "woba_value",
    ]

    suspicious: list[str] = []
    text_fields: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            text_fields.append(str(k))
            if isinstance(v, (list, tuple)):
                text_fields.extend([str(x) for x in v[:200]])
            elif isinstance(v, dict):
                text_fields.extend([str(x) for x in list(v.keys())[:200]])
            else:
                text_fields.append(str(v)[:500])
    for t in text_fields:
        tl = t.lower()
        if any(tok in tl for tok in tokens):
            suspicious.append(t)
    lines.append(f"Suspicious key/value string hits (sample): {suspicious[:200]}")

    feature_cols = []
    if isinstance(obj, dict):
        for k in ["feature_cols", "numeric_cols", "categorical_cols", "numeric_features", "categorical_features"]:
            v = obj.get(k)
            if isinstance(v, (list, tuple)):
                feature_cols.extend([str(x) for x in v])
    if feature_cols:
        leak_feats = [f for f in feature_cols if any(tok in f.lower() for tok in tokens)]
        lines.append(f"feature_cols_total={len(feature_cols)}")
        lines.append(f"feature_cols_suspicious={leak_feats}")
        if leak_feats:
            lines.append("WARNING: target-like/post-outcome columns found in feature columns.")
    else:
        lines.append("No explicit feature column list found in artifact.")

    lines.append("")
    lines.append("[5] PREDICTION RECONSTRUCTION TEST")
    lines.append("-" * 80)
    up = args.uncalibrated_draws_path.resolve()
    cpq = args.calibrated_draws_path.resolve()
    lines.append(f"uncalibrated_draws_path={up} exists={up.exists()}")
    lines.append(f"calibrated_draws_path={cpq} exists={cpq.exists()}")
    if up.exists() and cpq.exists():
        u = pd.read_parquet(up) if up.suffix.lower() == ".parquet" else pd.read_csv(up, low_memory=False)
        c = pd.read_parquet(cpq) if cpq.suffix.lower() == ".parquet" else pd.read_csv(cpq, low_memory=False)
        iu = infer_cols(u, is_cal=False)
        ic = infer_cols(c, is_cal=True)
        lines.append(f"inferred_uncal={iu}")
        lines.append(f"inferred_cal={ic}")
        eu = build_event_summary(u, iu, use_cal=False)
        ec = build_event_summary(c, ic, use_cal=True).rename(columns={"EV_target_mean": "EV_cal_mean"})
        evt = eu.merge(ec[["event_id", "EV_cal_mean"]], on="event_id", how="inner")
        evt["true_residual"] = evt["EV_obs"] - evt["EV_dec_mean"]
        evt["before_error"] = evt["EV_dec_mean"] - evt["EV_obs"]
        evt["actual_correction"] = evt["EV_cal_mean"] - evt["EV_dec_mean"]
        lines.append(f"event_rows={len(evt)}")
        lines.append(f"mean(true_residual)={fmt(evt['true_residual'].mean())}")
        lines.append(f"mean(before_error)={fmt(evt['before_error'].mean())}")
        lines.append(f"mean(actual_correction)={fmt(evt['actual_correction'].mean())}")

        predicted = None
        model = None
        if isinstance(obj, dict):
            model = obj.get("pipeline", None)
            if model is None:
                # maybe whole object is pipeline under other key
                for k, v in obj.items():
                    if hasattr(v, "predict"):
                        model = v
                        break
        elif hasattr(obj, "predict"):
            model = obj

        if model is not None and hasattr(model, "predict"):
            numf = []
            catf = []
            if isinstance(obj, dict):
                numf = list(obj.get("numeric_features", []))
                catf = list(obj.get("categorical_features", []))
            feats = numf + catf
            if not feats:
                cand = [c for c in evt.columns if c.startswith("EV_dec_") or c == "n_draws"]
                feats = cand
            X = evt.copy()
            for f in feats:
                if f not in X.columns:
                    X[f] = np.nan
            try:
                predicted = np.asarray(model.predict(X[feats]), dtype=float)
                evt["predicted"] = predicted
                lines.append("model.predict succeeded")
                lines.append(f"mean(predicted)={fmt(np.nanmean(predicted))}")
                lines.append(f"median(predicted)={fmt(np.nanmedian(predicted))}")
                lines.append(f"corr(predicted,true_residual)={fmt(safe_corr(predicted, evt['true_residual'].to_numpy()))}")
                lines.append(f"corr(predicted,actual_correction)={fmt(safe_corr(predicted, evt['actual_correction'].to_numpy()))}")
                lines.append(
                    f"RMSE(predicted,true_residual)={fmt(math.sqrt(mean_squared_error(evt['true_residual'], predicted)))}"
                )
                lines.append(
                    f"RMSE(predicted,actual_correction)={fmt(math.sqrt(mean_squared_error(evt['actual_correction'], predicted)))}"
                )
            except Exception as e:
                lines.append(f"model.predict failed: {e}")
        else:
            lines.append("No predict-capable model found.")

        if predicted is not None:
            mu_dec = evt["EV_dec_mean"].to_numpy(dtype=float)
            mu_cal = evt["EV_cal_mean"].to_numpy(dtype=float)
            pred = np.asarray(predicted, dtype=float)
            cap = float(obj.get("cap", 121.0)) if isinstance(obj, dict) else 121.0
            tau = float(obj.get("tau", 2.0)) if isinstance(obj, dict) else 2.0
            cands = {
                "A:dec_plus_pred": mu_dec + pred,
                "B:dec_minus_pred": mu_dec - pred,
                "C:pred_only": pred,
                "D:softcap(dec_plus_pred)": soft_cap(mu_dec + pred, cap=cap, tau=tau),
                "E:softcap(dec_minus_pred)": soft_cap(mu_dec - pred, cap=cap, tau=tau),
            }
            lines.append("")
            lines.append("Candidate formula diagnostics:")
            for name, arr in cands.items():
                absd = np.abs(mu_cal - arr)
                lines.append(
                    f"{name}: MAE_discrep={fmt(np.mean(absd))}, MedAE_discrep={fmt(np.median(absd))}, "
                    f"MaxAE_discrep={fmt(np.max(absd))}, corr_with_actual_EV_cal_mean={fmt(safe_corr(mu_cal, arr))}"
                )

    lines.append("")
    lines.append("[6] FINAL DIAGNOSIS")
    lines.append("-" * 80)
    if isinstance(obj, dict):
        has_pipeline = "pipeline" in obj and hasattr(obj["pipeline"], "predict")
        has_ev_meta = all(k in obj for k in ["rho", "cap", "tau", "ev_col", "group_col"])
        lines.append(f"artifact_is_dict=yes, has_pipeline={has_pipeline}, has_ev_meta={has_ev_meta}")
        if has_pipeline:
            lines.append("Artifact appears to store a residual prediction pipeline.")
        if not has_ev_meta:
            lines.append("Missing some metadata needed for reconstruction from artifact alone.")
    lines.append("Determine sign correctness from corr(predicted,true_residual) and formula MAE above.")
    lines.append("If B or E beats A/D by large margin, likely residual sign inversion.")
    lines.append("If C best, model may be predicting EV directly rather than residual.")
    lines.append("If softcap formulas fit much better than linear formulas, cap/shrinkage alters mean materially.")

    out = args.output_path.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

