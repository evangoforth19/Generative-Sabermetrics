#!/usr/bin/env python3
"""Plot AIC and BIC vs K from pitch_context_gmm_aic_bic.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/pitch_context_gmm/pitch_context_gmm_aic_bic.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/pitch_context_gmm/pitch_context_gmm_aic_bic_plots.png"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.rename(columns={"handedness": "hand"})
    # Normalize labels for legend
    label_map = {"L": "LHP", "R": "RHP"}
    df["hand_label"] = df["hand"].map(label_map).fillna(df["hand"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    colors = {"LHP": "#0d6e6e", "RHP": "#c45c26"}

    for ax, metric, title in zip(
        axes,
        ["aic", "bic"],
        ["AIC (lower is better)", "BIC (lower is better)"],
    ):
        for hand in ("LHP", "RHP"):
            sub = df[df["hand_label"] == hand]
            ax.plot(
                sub["k"],
                sub[metric],
                marker="o",
                ms=4,
                lw=2,
                label=hand,
                color=colors[hand],
            )
        ax.set_xlabel("Number of mixture components (K)")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(framealpha=0.95)

    fig.suptitle("Pitch-context GMM model selection", fontsize=13, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
