#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

HITTER_CANDIDATES = ["hitter", "player", "player_name", "batter_name", "hitter_name"]
CLUSTER_CANDIDATES = ["pitch_cluster", "cluster", "cluster_id", "pitch_cluster_id", "pitch_type_cluster", "cluster_global_id"]
VALUE_CANDIDATES = ["xwOBAcon", "xwobacon", "xwoba_con", "value", "value_xwobacon", "simulated_xwOBAcon"]

EXPECTED_HITTERS = [
    "Aaron Judge",
    "Mike Trout",
    "Giancarlo Stanton",
    "Pete Alonso",
    "Manny Machado",
    "Evan Longoria",
    "Luis Robert",
    "Mookie Betts",
    "George Springer",
    "Kris Bryant",
    "Nolan Arenado",
    "Alex Bregman",
]


def _norm_name(x: str) -> str:
    return str(x).strip().replace("_", " ").lower()


EXPECTED_NORM = {_norm_name(x): x for x in EXPECTED_HITTERS}


def _resolve_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer {label}. Tried {candidates}. Available columns: {list(df.columns)}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def _iter_input_files(input_dir: Path) -> Iterable[Path]:
    pats = ["**/*robustness_draws.parquet", "**/*robustness_draws.csv", "**/*.parquet", "**/*.csv"]
    seen: set[str] = set()
    for pat in pats:
        for p in sorted(input_dir.glob(pat)):
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield p


def _load_input(input_path: Path | None, input_dir: Path | None) -> tuple[pd.DataFrame, list[str]]:
    if input_path is None and input_dir is None:
        raise ValueError("Provide either --input-path or --input-dir")
    warnings: list[str] = []
    if input_path is not None and input_dir is not None:
        warnings.append("Both --input-path and --input-dir provided; using --input-path.")
    if input_path is not None:
        return _read_table(input_path), warnings
    assert input_dir is not None
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    parts: list[pd.DataFrame] = []
    for p in _iter_input_files(input_dir):
        try:
            dfp = _read_table(p)
        except Exception:
            continue
        parts.append(dfp)
    if not parts:
        raise FileNotFoundError(f"No csv/parquet files found in {input_dir}")
    return pd.concat(parts, ignore_index=True, sort=False), warnings


def _save_plots(cluster_means: pd.DataFrame, hitter_metrics: pd.DataFrame, out_dir: Path) -> None:
    # 1) robustness bar
    r = hitter_metrics.sort_values("pitch_type_robustness_score", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(r["hitter"], r["pitch_type_robustness_score"])
    ax.set_title("Pitch-Type Robustness Score (higher better)")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "pitch_type_robustness_score_bar.png", dpi=130)
    plt.close(fig)

    # 2) vulnerability gap bar
    v = hitter_metrics.sort_values("vulnerability_gap", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(v["hitter"], v["vulnerability_gap"])
    ax.set_title("Vulnerability Gap (lower better)")
    ax.set_ylabel("Gap")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "vulnerability_gap_bar.png", dpi=130)
    plt.close(fig)

    # 3) mean vs floor scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hitter_metrics["mean_pitch_value"], hitter_metrics["pitch_type_robustness_score"], s=50)
    for _, rr in hitter_metrics.iterrows():
        ax.annotate(rr["hitter"], (rr["mean_pitch_value"], rr["pitch_type_robustness_score"]), fontsize=8, alpha=0.9)
    ax.set_xlabel("Mean Pitch Value")
    ax.set_ylabel("Pitch-Type Robustness Score")
    ax.set_title("Mean vs Robustness Floor")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_vs_floor_scatter.png", dpi=130)
    plt.close(fig)

    # 4) mean vs vulnerability gap
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hitter_metrics["mean_pitch_value"], hitter_metrics["vulnerability_gap"], s=50)
    for _, rr in hitter_metrics.iterrows():
        ax.annotate(rr["hitter"], (rr["mean_pitch_value"], rr["vulnerability_gap"]), fontsize=8, alpha=0.9)
    ax.set_xlabel("Mean Pitch Value")
    ax.set_ylabel("Vulnerability Gap")
    ax.set_title("Mean vs Vulnerability Gap")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_vs_vulnerability_gap_scatter.png", dpi=130)
    plt.close(fig)

    # 5) cluster mean distribution boxplot
    hitters = list(hitter_metrics.sort_values("hitter")["hitter"])
    data = [cluster_means.loc[cluster_means["hitter"].eq(h), "mean_xwOBAcon"].to_numpy(dtype=float) for h in hitters]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=hitters, showfliers=False)
    ax.set_title("Distribution of Cluster Mean xwOBAcon by Hitter")
    ax.set_ylabel("Cluster Mean xwOBAcon")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_mean_distributions_boxplot.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", type=Path, default=None)
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path(__file__).resolve().parents[1] / "outputs" / "pitch_type_robustness" / stamp)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df, warnings = _load_input(args.input_path, args.input_dir)
    hitter_col = _resolve_col(df, HITTER_CANDIDATES, "hitter column")
    cluster_col = _resolve_col(df, CLUSTER_CANDIDATES, "pitch cluster column")
    value_col = _resolve_col(df, VALUE_CANDIDATES, "value/xwOBAcon column")

    missing_hitter = int(df[hitter_col].isna().sum())
    missing_cluster = int(df[cluster_col].isna().sum())
    missing_value = int(pd.to_numeric(df[value_col], errors="coerce").isna().sum())
    if missing_hitter > 0:
        warnings.append(f"Missing hitter values: {missing_hitter}")
    if missing_cluster > 0:
        warnings.append(f"Missing cluster values: {missing_cluster}")
    if missing_value > 0:
        warnings.append(f"Missing value rows before drop: {missing_value}")

    d = df.copy()
    d = d.dropna(subset=[hitter_col, cluster_col]).copy()
    d["value_num"] = pd.to_numeric(d[value_col], errors="coerce")
    dropped_val = int(d["value_num"].isna().sum())
    if dropped_val > 0:
        warnings.append(f"Dropped rows with missing value: {dropped_val}")
    d = d.dropna(subset=["value_num"]).copy()

    d["hitter"] = d[hitter_col].astype(str).map(lambda x: EXPECTED_NORM.get(_norm_name(x), str(x).strip().replace("_", " ").title()))
    d["pitch_cluster"] = d[cluster_col]

    cluster_means = (
        d.groupby(["hitter", "pitch_cluster"], as_index=False)
        .agg(
            mean_xwOBAcon=("value_num", "mean"),
            std_xwOBAcon=("value_num", lambda s: float(np.std(s, ddof=0))),
            n_draws=("value_num", "size"),
        )
    )
    cluster_means["cluster_rank_within_hitter"] = (
        cluster_means.groupby("hitter")["mean_xwOBAcon"].rank(method="first", ascending=True).astype(int)
    )

    hitter_rows: list[dict] = []
    for h, g in cluster_means.groupby("hitter", sort=True):
        g2 = g.sort_values("mean_xwOBAcon", ascending=True).reset_index(drop=True)
        n_clusters = int(g2["pitch_cluster"].nunique())
        if n_clusters != 57:
            warnings.append(f"Hitter {h}: expected 57 clusters, found {n_clusters}")
        if int((g2["n_draws"] < 25).sum()) > 0:
            warnings.append(f"Hitter {h}: {int((g2['n_draws'] < 25).sum())} cluster(s) with very few draws (<25)")
        bottom_k = int(math.ceil(0.10 * max(n_clusters, 1)))
        bottom_k = max(bottom_k, 1)
        top_k = bottom_k

        mu_bar = float(g2["mean_xwOBAcon"].mean())
        floor = float(g2["mean_xwOBAcon"].head(bottom_k).mean())
        vuln = float(mu_bar - floor)
        robust = float(0.5 * mu_bar + 0.5 * floor)

        hitter_rows.append(
            {
                "hitter": h,
                "n_clusters": n_clusters,
                "total_draws": int(g2["n_draws"].sum()),
                "mean_pitch_value": mu_bar,
                "pitch_type_robustness_score": floor,
                "vulnerability_gap": vuln,
                "robust_value_score": robust,
                "worst_cluster_mean": float(g2["mean_xwOBAcon"].min()),
                "best_cluster_mean": float(g2["mean_xwOBAcon"].max()),
                "cluster_mean_std": float(np.std(g2["mean_xwOBAcon"].to_numpy(dtype=float), ddof=0)),
                "cluster_mean_range": float(g2["mean_xwOBAcon"].max() - g2["mean_xwOBAcon"].min()),
                "bottom_k": bottom_k,
                "bottom_clusters": ",".join(map(str, g2["pitch_cluster"].head(bottom_k).tolist())),
                "best_clusters": ",".join(map(str, g2["pitch_cluster"].tail(top_k).tolist())),
            }
        )

    hitter_metrics = pd.DataFrame(hitter_rows)
    hitter_metrics["pitch_type_robustness_rank"] = (
        hitter_metrics["pitch_type_robustness_score"].rank(method="min", ascending=False).astype(int)
    )
    hitter_metrics["vulnerability_gap_rank"] = (
        hitter_metrics["vulnerability_gap"].rank(method="min", ascending=True).astype(int)
    )
    hitter_metrics["robust_value_rank"] = (
        hitter_metrics["robust_value_score"].rank(method="min", ascending=False).astype(int)
    )

    round_cols = [
        "mean_pitch_value",
        "pitch_type_robustness_score",
        "vulnerability_gap",
        "robust_value_score",
        "worst_cluster_mean",
        "best_cluster_mean",
        "cluster_mean_std",
        "cluster_mean_range",
        "mean_xwOBAcon",
        "std_xwOBAcon",
    ]
    cluster_means_out = cluster_means.copy()
    for c in [x for x in round_cols if x in cluster_means_out.columns]:
        cluster_means_out[c] = cluster_means_out[c].round(4)
    hitter_metrics_out = hitter_metrics.copy()
    for c in [x for x in round_cols if x in hitter_metrics_out.columns]:
        hitter_metrics_out[c] = hitter_metrics_out[c].round(4)

    cluster_means_out.to_csv(out_dir / "hitter_pitch_cluster_means.csv", index=False)
    hitter_metrics_out.to_csv(out_dir / "hitter_robustness_metrics.csv", index=False)
    hitter_metrics_out.sort_values("pitch_type_robustness_score", ascending=False).to_csv(
        out_dir / "pitch_type_robustness_rankings.csv", index=False
    )
    hitter_metrics_out.sort_values("vulnerability_gap", ascending=True).to_csv(
        out_dir / "vulnerability_gap_rankings.csv", index=False
    )

    _save_plots(cluster_means, hitter_metrics, out_dir)

    top_rob = hitter_metrics.sort_values("pitch_type_robustness_score", ascending=False).iloc[0]
    top_vgap = hitter_metrics.sort_values("vulnerability_gap", ascending=True).iloc[0]
    top_rvs = hitter_metrics.sort_values("robust_value_score", ascending=False).iloc[0]

    missing_expected = [h for h in EXPECTED_HITTERS if h not in set(hitter_metrics["hitter"])]
    if missing_expected:
        warnings.append(f"Expected hitters missing from processed output: {missing_expected}")

    lines = []
    lines.append("Pitch-Type Robustness Report")
    lines.append("============================")
    lines.append(f"Hitters processed: {hitter_metrics.shape[0]}")
    lines.append(f"Bottom-k rule: ceil(0.10 * n_clusters) per hitter")
    lines.append("")
    lines.append(f"Top by Pitch-Type Robustness Score: {top_rob['hitter']} ({top_rob['pitch_type_robustness_score']:.4f})")
    lines.append(f"Top by lowest Vulnerability Gap: {top_vgap['hitter']} ({top_vgap['vulnerability_gap']:.4f})")
    lines.append(f"Top by Robust Value Score: {top_rvs['hitter']} ({top_rvs['robust_value_score']:.4f})")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- mean_pitch_value: average cluster mean value across all pitch designs.")
    lines.append("- pitch_type_robustness_score: mean of bottom 10% cluster means (downside floor); higher is better.")
    lines.append("- vulnerability_gap: average minus floor; lower means less exploitable downside.")
    lines.append("")
    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("Warnings: none")
    (out_dir / "robustness_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    r1 = hitter_metrics.sort_values("pitch_type_robustness_score", ascending=False).reset_index(drop=True)
    r2 = hitter_metrics.sort_values("vulnerability_gap", ascending=True).reset_index(drop=True)
    print("Pitch-Type Robustness Rankings")
    print("------------------------------")
    for i, rr in r1.iterrows():
        print(f"{i+1}. {rr['hitter']} ({rr['pitch_type_robustness_score']:.4f})")
    print("")
    print("Vulnerability Gap Rankings")
    print("--------------------------")
    for i, rr in r2.iterrows():
        print(f"{i+1}. {rr['hitter']} ({rr['vulnerability_gap']:.4f})")
    print("")
    print("Saved outputs to:")
    print(str(out_dir))


if __name__ == "__main__":
    main()

