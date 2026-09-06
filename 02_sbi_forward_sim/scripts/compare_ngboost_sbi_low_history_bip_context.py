#!/usr/bin/env python3
"""
Compare NGBoost (context-only), Deep Ensemble MLP, and SBI forward + skew-normal EV refit on
held-out events where the batter has sparse training history for the same (z_count, pitch_type)
context.

Support count: number of rows in baseline_direct_y_train + baseline_direct_y_calibration with
matching (batter_name, z_count, pitch_type). Each row is a training BIP / direct-y example.

Metrics for EV: MAE & RMSE use the model predictive mean (NGBoost/MLP: pred_mean_EV; SBI:
skew-normal fitted mean). NLL / CRPS / coverage match the respective evaluation scripts (Gaussian
for black-box baselines, skew-normal for SBI refit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["batter_name", "z_count", "pitch_type"]


def coverage_interval_normal(
    y: np.ndarray, mu: np.ndarray, sig: np.ndarray, q_lo: float, q_hi: float
) -> np.ndarray:
    sig = np.maximum(sig.astype(np.float64), 1e-8)
    lo = stats.norm.ppf(q_lo, loc=mu, scale=sig)
    hi = stats.norm.ppf(q_hi, loc=mu, scale=sig)
    return ((lo <= y) & (y <= hi)).astype(np.float64)


def agg_gaussian_ev(df: pd.DataFrame) -> dict[str, float]:
    y = df["obs_EV"].to_numpy(dtype=np.float64)
    mu = df["pred_mean_EV"].to_numpy(dtype=np.float64)
    sig = np.maximum(df["pred_std_EV"].to_numpy(dtype=np.float64), 1e-8)
    return {
        "n_events": int(len(df)),
        "MAE": float(np.mean(np.abs(mu - y))),
        "RMSE": float(np.sqrt(np.mean((mu - y) ** 2))),
        "NLL": float(np.nanmean(df["nll_EV"].to_numpy())),
        "COV50": float(coverage_interval_normal(y, mu, sig, 0.25, 0.75).mean()),
        "COV80": float(coverage_interval_normal(y, mu, sig, 0.10, 0.90).mean()),
        "COV90": float(coverage_interval_normal(y, mu, sig, 0.05, 0.95).mean()),
        "CRPS": float(np.nanmean(df["crps_EV"].to_numpy())),
    }


def agg_sbi_skew(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n_events": int(len(df)),
        "MAE": float(df["abs_error"].mean()),
        "RMSE": float(np.sqrt(df["sq_error"].mean())),
        "NLL": float(df["nll"].mean()),
        "COV50": float(df["cov50"].mean()),
        "COV80": float(df["cov80"].mean()),
        "COV90": float(df["cov90"].mean()),
        "CRPS": float(df["crps"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-train",
        type=Path,
        default=Path("data_processed/baseline_direct_y_train.parquet"),
    )
    ap.add_argument(
        "--baseline-cal",
        type=Path,
        default=Path("data_processed/baseline_direct_y_calibration.parquet"),
    )
    ap.add_argument(
        "--baseline-test",
        type=Path,
        default=Path("data_processed/baseline_direct_y_test.parquet"),
    )
    ap.add_argument(
        "--ngboost-scores",
        type=Path,
        required=True,
        help="per_event_predictive_scores_ngboost_style.csv from black_box_baselines",
    )
    ap.add_argument(
        "--sbi-skew-per-event",
        type=Path,
        required=True,
        help="ev_skewnormal_refit_per_event.csv from SBI held-out run",
    )
    ap.add_argument(
        "--mlp-scores",
        type=Path,
        default=None,
        help="per_event_predictive_scores_deep_ensemble.csv (optional)",
    )
    ap.add_argument("--support-threshold", type=int, default=10)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_p = (root / args.baseline_train).resolve()
    cal_p = (root / args.baseline_cal).resolve()
    test_p = (root / args.baseline_test).resolve()
    for p in (train_p, cal_p, test_p):
        if not p.is_file():
            raise FileNotFoundError(p)

    hist = pd.concat(
        [pd.read_parquet(train_p, columns=KEYS), pd.read_parquet(cal_p, columns=KEYS)],
        ignore_index=True,
    )
    support = hist.groupby(KEYS, dropna=False).size().rename("n_hist_train_cal_bip").reset_index()

    test = pd.read_parquet(test_p, columns=["event_id"] + KEYS)
    test = test.drop_duplicates("event_id", keep="first")
    m = test.merge(support, on=KEYS, how="left")
    m["n_hist_train_cal_bip"] = m["n_hist_train_cal_bip"].fillna(0).astype(int)
    thr = int(args.support_threshold)
    m["low_history"] = m["n_hist_train_cal_bip"] < thr

    ng = pd.read_csv(args.ngboost_scores.resolve())
    sk = pd.read_csv(args.sbi_skew_per_event.resolve())
    if set(ng["event_id"]) != set(sk["event_id"]):
        raise ValueError("event_id mismatch between NGBoost and SBI skew tables")

    base = m.merge(ng, on="event_id", how="inner").merge(sk, on="event_id", how="inner", suffixes=("_ng", "_sk"))
    mlp_path = args.mlp_scores.resolve() if args.mlp_scores is not None else None
    if mlp_path is not None:
        if not mlp_path.is_file():
            raise FileNotFoundError(mlp_path)
        mlp = pd.read_csv(mlp_path)
        if set(mlp["event_id"]) != set(base["event_id"]):
            raise ValueError("event_id mismatch between MLP and other tables")
        mlp_ev = mlp[
            ["event_id", "obs_EV", "pred_mean_EV", "pred_std_EV", "nll_EV", "crps_EV"]
        ].rename(
            columns={
                "obs_EV": "obs_EV_mlp",
                "pred_mean_EV": "pred_mean_EV_mlp",
                "pred_std_EV": "pred_std_EV_mlp",
                "nll_EV": "nll_EV_mlp",
                "crps_EV": "crps_EV_mlp",
            }
        )
        base = base.merge(mlp_ev, on="event_id", how="inner")

    low = base.loc[base["low_history"]].copy()
    high = base.loc[~base["low_history"]].copy()

    def agg_mlp_slice(df: pd.DataFrame) -> dict[str, float]:
        sub = pd.DataFrame(
            {
                "obs_EV": df["obs_EV_mlp"],
                "pred_mean_EV": df["pred_mean_EV_mlp"],
                "pred_std_EV": df["pred_std_EV_mlp"],
                "nll_EV": df["nll_EV_mlp"],
                "crps_EV": df["crps_EV_mlp"],
            }
        )
        return agg_gaussian_ev(sub)

    def agg_ng_slice(df: pd.DataFrame) -> dict[str, float]:
        sub = pd.DataFrame(
            {
                "obs_EV": df["obs_EV_ng"] if "obs_EV_ng" in df.columns else df["obs_EV"],
                "pred_mean_EV": df["pred_mean_EV_ng"] if "pred_mean_EV_ng" in df.columns else df["pred_mean_EV"],
                "pred_std_EV": df["pred_std_EV_ng"] if "pred_std_EV_ng" in df.columns else df["pred_std_EV"],
                "nll_EV": df["nll_EV_ng"] if "nll_EV_ng" in df.columns else df["nll_EV"],
                "crps_EV": df["crps_EV_ng"] if "crps_EV_ng" in df.columns else df["crps_EV"],
            }
        )
        return agg_gaussian_ev(sub)

    out = {
        "definition": {
            "support_keys": KEYS,
            "support_count_source": "Rows in baseline_direct_y_train + baseline_direct_y_calibration.",
            "low_history": f"n_hist_train_cal_bip < {thr}",
            "ngboost_scores": str(args.ngboost_scores.resolve()),
            "sbi_skew_refit": str(args.sbi_skew_per_event.resolve()),
            "mlp_scores": str(mlp_path) if mlp_path is not None else None,
        },
        "counts": {
            "held_out_events_total": int(len(base)),
            "low_history_events": int(low["event_id"].nunique()),
            "high_history_events": int(high["event_id"].nunique()),
        },
        "low_history": {
            "ngboost": agg_ng_slice(low),
            "sbi_skewnorm_250_adm": agg_sbi_skew(low),
        },
        "high_history": {
            "ngboost": agg_ng_slice(high),
            "sbi_skewnorm_250_adm": agg_sbi_skew(high),
        },
        "all_events": {
            "ngboost": agg_ng_slice(base),
            "sbi_skewnorm_250_adm": agg_sbi_skew(base),
        },
    }
    if mlp_path is not None:
        out["low_history"]["deep_ensemble_mlp"] = agg_mlp_slice(low)
        out["high_history"]["deep_ensemble_mlp"] = agg_mlp_slice(high)
        out["all_events"]["deep_ensemble_mlp"] = agg_mlp_slice(base)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def table_block(title: str, models: dict[str, dict]) -> list[str]:
        names = list(models.keys())
        header = "| Metric | " + " | ".join(names) + " |"
        sep = "|--------|" + "|".join(["--------:"] * len(names)) + "|"
        lines = [f"### {title}", "", header, sep]
        for k in ["MAE", "RMSE", "NLL", "COV50", "COV80", "COV90", "CRPS"]:
            vals = " | ".join(f"{models[n][k]:.6f}" for n in names)
            lines.append(f"| {k} | {vals} |")
        lines.append("")
        lines.append(f"- **n events:** {next(iter(models.values()))['n_events']}")
        lines.append("")
        return lines

    def models_for_slice(key: str) -> dict[str, dict]:
        d = {
            "NGBoost": out[key]["ngboost"],
            "SBI skew-normal (250 adm)": out[key]["sbi_skewnorm_250_adm"],
        }
        if "deep_ensemble_mlp" in out[key]:
            d["Deep Ensemble MLP"] = out[key]["deep_ensemble_mlp"]
        return d

    if thr == 1:
        low_def = (
            "**Zero-history definition:** **exactly 0** training+calibration BIP rows for the same "
            "`(batter_name, z_count, pitch_type)` as the held-out pitch "
            "(equivalently: count `< 1`)."
        )
        low_heading = "## Zero-history subset (0 train+cal BIP in context)"
    else:
        low_def = (
            f"**Low-history definition:** fewer than **{thr}** training+calibration BIP rows for the same "
            "`(batter_name, z_count, pitch_type)` as the held-out pitch."
        )
        low_heading = f"## Low-history subset (< {thr} training+cal BIP in context)"

    md = [
        "# NGBoost vs SBI forward (skew-normal on 250 admissible draws): history-stratified EV comparison",
        "",
        low_def,
        "",
        f"- Held-out events (intersection of tables): **{out['counts']['held_out_events_total']}**",
        f"- Low-history subset: **{out['counts']['low_history_events']}**",
        f"- Remaining (history ≥ {thr}): **{out['counts']['high_history_events']}**",
        "",
        low_heading,
        "",
    ]
    md.extend(table_block("Metrics (EV)", models_for_slice("low_history")))
    md.extend(["---", "", f"## High-history subset (≥ {thr} BIP in context)", ""])
    md.extend(table_block("Metrics (EV)", models_for_slice("high_history")))
    md.extend(["---", "", "## All held-out events (reference)", ""])
    md.extend(table_block("Metrics (EV)", models_for_slice("all_events")))
    md.extend(
        [
            "---",
            "",
            "## Files",
            "",
            f"- JSON: `{args.out_json}`",
            "",
        ]
    )
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"ok": True, "out_json": str(args.out_json), "out_md": str(args.out_md), "summary": out["counts"]}, indent=2))


if __name__ == "__main__":
    main()
