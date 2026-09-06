#!/usr/bin/env python3
"""
Build presentation-style EV diagnostic tables (Generative vs NGBoost vs Deep Ensemble MLP)
with a Winner row, plus JSON/markdown and PNG/PDF figures.

Metrics: RMSE, NLL, COV50, COV80, COV90, CRPS (nan-mean over events; EV only).
Generative = per-event skew-normal refit on admissible forward-simulator draws.
NGBoost / MLP = context-only Gaussian marginals from black_box_baselines per-event CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
_KEYS = ["batter_name", "z_count", "pitch_type"]
_METRICS = ["RMSE", "NLL", "COV50", "COV80", "COV90", "CRPS"]
_NOMINAL = {"COV50": 0.5, "COV80": 0.8, "COV90": 0.9}
_LOWER_BETTER = frozenset({"RMSE", "NLL", "CRPS"})

_MODEL_ORDER = ("generative", "ngboost", "deep_ensemble_mlp")
_DISPLAY = {
    "generative": "Generative Model",
    "ngboost": "NGBoost-Style Baseline",
    "deep_ensemble_mlp": "Deep Ensemble MLP",
}

_NGB_DEFAULT = _ROOT / "outputs/black_box_baselines/20260408_044530Z/per_event_predictive_scores_ngboost_style.csv"
_MLP_DEFAULT = _ROOT / "outputs/black_box_baselines/20260408_044530Z/per_event_predictive_scores_deep_ensemble.csv"
_RUN_DEFAULT = _ROOT / "outputs/heldout_pipeline_test/20260409_175049Z"


def coverage_interval_normal(
    y: np.ndarray, mu: np.ndarray, sig: np.ndarray, q_lo: float, q_hi: float
) -> np.ndarray:
    sig = np.maximum(sig.astype(np.float64), 1e-8)
    lo = stats.norm.ppf(q_lo, loc=mu, scale=sig)
    hi = stats.norm.ppf(q_hi, loc=mu, scale=sig)
    return ((lo <= y) & (y <= hi)).astype(np.float64)


def agg_gaussian_black_box(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["obs_EV"].to_numpy(dtype=np.float64)
    mu = df["pred_mean_EV"].to_numpy(dtype=np.float64)
    sig = np.maximum(df["pred_std_EV"].to_numpy(dtype=np.float64), 1e-8)
    return {
        "n_events": int(len(df)),
        "RMSE": float(np.sqrt(np.mean((mu - y) ** 2))),
        "NLL": float(np.nanmean(df["nll_EV"].to_numpy())),
        "COV50": float(coverage_interval_normal(y, mu, sig, 0.25, 0.75).mean()),
        "COV80": float(coverage_interval_normal(y, mu, sig, 0.10, 0.90).mean()),
        "COV90": float(coverage_interval_normal(y, mu, sig, 0.05, 0.95).mean()),
        "CRPS": float(np.nanmean(df["crps_EV"].to_numpy())),
    }


def agg_generative_skew(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n_events": int(len(df)),
        "RMSE": float(np.sqrt(df["sq_error"].mean())),
        "NLL": float(df["nll"].mean()),
        "COV50": float(df["cov50"].mean()),
        "COV80": float(df["cov80"].mean()),
        "COV90": float(df["cov90"].mean()),
        "CRPS": float(df["crps"].mean()),
    }


def pick_winner(metric: str, by_model: dict[str, dict[str, float | int]]) -> str:
    vals = {m: float(by_model[m][metric]) for m in _MODEL_ORDER}
    if metric in _LOWER_BETTER:
        best_v = min(vals.values())
        winners = [m for m, v in vals.items() if v == best_v]
    else:
        nom = _NOMINAL[metric]
        errs = {m: abs(vals[m] - nom) for m in _MODEL_ORDER}
        best_e = min(errs.values())
        winners = [m for m, e in errs.items() if e == best_e]
    if len(winners) == 1:
        return _DISPLAY[winners[0]]
    return " / ".join(_DISPLAY[w] for w in winners)


def aggregate_slice(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    gen_df = df[["sq_error", "nll", "cov50", "cov80", "cov90", "crps"]]
    ng_df = pd.DataFrame(
        {
            "obs_EV": df["obs_EV_ng"],
            "pred_mean_EV": df["pred_mean_EV_ng"],
            "pred_std_EV": df["pred_std_EV_ng"],
            "nll_EV": df["nll_EV_ng"],
            "crps_EV": df["crps_EV_ng"],
        }
    )
    mlp_df = pd.DataFrame(
        {
            "obs_EV": df["obs_EV_mlp"],
            "pred_mean_EV": df["pred_mean_EV_mlp"],
            "pred_std_EV": df["pred_std_EV_mlp"],
            "nll_EV": df["nll_EV_mlp"],
            "crps_EV": df["crps_EV_mlp"],
        }
    )
    return {
        "generative": agg_generative_skew(gen_df),
        "ngboost": agg_gaussian_black_box(ng_df),
        "deep_ensemble_mlp": agg_gaussian_black_box(mlp_df),
    }


def _merge_scores(
    event_meta: pd.DataFrame,
    ng: pd.DataFrame,
    mlp: pd.DataFrame,
    sk: pd.DataFrame,
) -> pd.DataFrame:
    ng_ev = ng[
        ["event_id", "obs_EV", "pred_mean_EV", "pred_std_EV", "nll_EV", "crps_EV"]
    ].rename(
        columns={
            "obs_EV": "obs_EV_ng",
            "pred_mean_EV": "pred_mean_EV_ng",
            "pred_std_EV": "pred_std_EV_ng",
            "nll_EV": "nll_EV_ng",
            "crps_EV": "crps_EV_ng",
        }
    )
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
    sk_ev = sk[
        [
            "event_id",
            "sq_error",
            "nll",
            "cov50",
            "cov80",
            "cov90",
            "crps",
        ]
    ]
    base = event_meta.merge(ng_ev, on="event_id").merge(mlp_ev, on="event_id").merge(sk_ev, on="event_id")
    return base


def render_table_figure(
    title: str,
    by_model: dict[str, dict[str, float | int]],
    n_events: int,
    out_png: Path,
    out_pdf: Path | None,
) -> None:
    col_labels = ["Model", "n_events"] + list(_METRICS)
    row_labels = [_DISPLAY[m] for m in _MODEL_ORDER] + ["Winner"]
    nrows = len(row_labels)
    ncols = len(col_labels)
    data: list[list[str]] = []
    for m in _MODEL_ORDER:
        row = [_DISPLAY[m], f"{n_events:.0f}"]
        for k in _METRICS:
            row.append(f"{by_model[m][k]:.2f}")
        data.append(row)
    win_row = ["Winner", "—"]
    for k in _METRICS:
        win_row.append(pick_winner(k, by_model))
    data.append(win_row)

    fig_w = max(10.0, 1.15 * ncols)
    fig_h = 0.55 * nrows + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif r == nrows:
            cell.set_facecolor("#E7E6E6")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=13, weight="bold", pad=16)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    if out_pdf is not None:
        with PdfPages(out_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(
    title: str,
    by_model: dict[str, dict[str, float | int]],
    n_events: int,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Model | n_events | " + " | ".join(_METRICS) + " |",
        "|-------|--------:|" + "|".join(["--------:"] * len(_METRICS)) + "|",
    ]
    for m in _MODEL_ORDER:
        vals = " | ".join(f"{by_model[m][k]:.6f}" for k in _METRICS)
        lines.append(f"| {_DISPLAY[m]} | {n_events} | {vals} |")
    win_cells = " | ".join(pick_winner(k, by_model) for k in _METRICS)
    lines.append(f"| **Winner** | — | {win_cells} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-run", type=Path, default=_RUN_DEFAULT)
    ap.add_argument("--ngboost-scores", type=Path, default=_NGB_DEFAULT)
    ap.add_argument("--mlp-scores", type=Path, default=_MLP_DEFAULT)
    ap.add_argument(
        "--sbi-skew-per-event",
        type=Path,
        default=None,
        help="Default: <heldout-run>/ev_skewnormal_refit_per_event.csv",
    )
    ap.add_argument("--support-threshold", type=int, default=3)
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Default: <heldout-run>/plots/model_diagnostics",
    )
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    run_dir = (_ROOT / args.heldout_run).resolve() if not args.heldout_run.is_absolute() else args.heldout_run.resolve()
    sk_path = args.sbi_skew_per_event or (run_dir / "ev_skewnormal_refit_per_event.csv")
    sk_path = sk_path.resolve()
    ng_path = args.ngboost_scores.resolve()
    mlp_path = args.mlp_scores.resolve()
    plots_dir = args.plots_dir or (run_dir / "plots" / "model_diagnostics")
    plots_dir = plots_dir.resolve()

    train_p = _ROOT / "data_processed/baseline_direct_y_train.parquet"
    cal_p = _ROOT / "data_processed/baseline_direct_y_calibration.parquet"
    test_p = _ROOT / "data_processed/baseline_direct_y_test.parquet"
    for p in (train_p, cal_p, test_p, sk_path, ng_path, mlp_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    hist = pd.concat(
        [pd.read_parquet(train_p, columns=_KEYS), pd.read_parquet(cal_p, columns=_KEYS)],
        ignore_index=True,
    )
    support = hist.groupby(_KEYS, dropna=False).size().rename("n_hist_train_cal_bip").reset_index()
    test = pd.read_parquet(test_p, columns=["event_id"] + _KEYS).drop_duplicates("event_id", keep="first")
    meta = test.merge(support, on=_KEYS, how="left")
    meta["n_hist_train_cal_bip"] = meta["n_hist_train_cal_bip"].fillna(0).astype(int)
    thr = int(args.support_threshold)
    meta["low_history"] = meta["n_hist_train_cal_bip"] < thr

    ng = pd.read_csv(ng_path)
    mlp = pd.read_csv(mlp_path)
    sk = pd.read_csv(sk_path)
    eids = set(ng["event_id"].astype(int))
    if eids != set(mlp["event_id"].astype(int)) or eids != set(sk["event_id"].astype(int)):
        raise ValueError("event_id mismatch among NGBoost, MLP, and generative skew tables")

    base = _merge_scores(meta, ng, mlp, sk)
    low = base.loc[base["low_history"]].copy()
    high = base.loc[~base["low_history"]].copy()

    slices = {
        "all_events": (base, "Model Diagnostic Results (EV, all held-out events)"),
        f"low_history_lt{thr}": (
            low,
            f"Low-history subset (< {thr} training+cal BIP in context): EV Metrics",
        ),
        f"high_history_ge{thr}": (
            high,
            f"High-history subset (≥ {thr} training+cal BIP in context): EV Metrics",
        ),
    }

    out_payload: dict = {
        "heldout_run": str(run_dir),
        "support_threshold": thr,
        "inputs": {
            "ngboost_scores": str(ng_path),
            "mlp_scores": str(mlp_path),
            "sbi_skew_per_event": str(sk_path),
        },
        "counts": {
            "held_out_events_total": int(len(base)),
            "low_history_events": int(len(low)),
            "high_history_events": int(len(high)),
        },
        "slices": {},
        "figures": {},
    }

    md_lines = [
        "# EV model diagnostic comparison (Generative vs NGBoost vs Deep Ensemble MLP)",
        "",
        f"- **Held-out run:** `{run_dir.name}`",
        f"- **Low-history:** `n_hist_train_cal_bip < {thr}` on `{_KEYS}`",
        f"- **Events:** {len(base)} total, {len(low)} low-history, {len(high)} high-history",
        "",
    ]

    for key, (df_slice, plot_title) in slices.items():
        agg = aggregate_slice(df_slice)
        n_ev = int(agg["generative"]["n_events"])
        out_payload["slices"][key] = agg

        stem = f"model_diagnostic_{key}"
        png_path = plots_dir / f"{stem}.png"
        pdf_path = None if args.no_pdf else plots_dir / f"{stem}.pdf"
        render_table_figure(plot_title, agg, n_ev, png_path, pdf_path)
        out_payload["figures"][key] = {"png": str(png_path), "pdf": str(pdf_path) if pdf_path else None}

        md_lines.extend(markdown_table(plot_title, agg, n_ev))

    json_path = run_dir / "model_diagnostic_comparison_3way.json"
    md_path = run_dir / "model_diagnostic_comparison_3way.md"
    json_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "json": str(json_path),
                "markdown": str(md_path),
                "plots_dir": str(plots_dir),
                "figures": out_payload["figures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
