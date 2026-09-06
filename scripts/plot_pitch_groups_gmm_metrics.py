#!/usr/bin/env python3
"""Plot AIC/BIC vs K for each pitch_group from pitch_groups_gmm_aic_bic.csv."""

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
        default=Path("outputs/pitch_context_gmm_pitch_groups/pitch_groups_gmm_aic_bic.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/pitch_context_gmm_pitch_groups/pitch_groups_gmm_aic_bic_plots.png"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    order = ["4F", "2F", "CF", "S", "C", "CH"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes_flat = axes.ravel()

    for ax, grp in zip(axes_flat, order):
        sub = df[df["pitch_group"] == grp]
        ax.plot(sub["k"], sub["aic"], marker="o", ms=3, lw=1.8, label="AIC", color="#1f77b4")
        ax.plot(sub["k"], sub["bic"], marker="s", ms=3, lw=1.8, label="BIC", color="#ff7f0e")
        ax.set_title(f"Pitch group {grp}")
        ax.set_xlabel("K")
        ax.set_ylabel("information criterion")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Pitch-context GMM: AIC & BIC by pitch-type group", fontsize=13, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
