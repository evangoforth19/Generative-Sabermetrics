#!/usr/bin/env python3
"""
Per-hitter marginal density plots for EV, LA, and SA (simulated vs observed BIP).

Writes four PNGs per player under the same folder as the simulation outputs:
  ``{slug}_marginal_ev.png``, ``{slug}_marginal_la.png``, ``{slug}_marginal_sa.png``,
  and ``{slug}_marginal_value.png`` (xwOBAcon from the value model; simulated vs observed BIP).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO / "outputs/sbi_bootstrap_value_per_hitter_fullrun"
DEFAULT_RUN_ROOT_TRUNC_EX = REPO / "outputs/sbi_bootstrap_value_per_hitter_trunc_ex_vm"


def _title_slug(slug: str) -> str:
    return re.sub(r"\s+", " ", slug.replace("_", " ").strip()).title()


def _kde_x(xs: np.ndarray, n_grid: int = 256) -> tuple[np.ndarray, np.ndarray] | None:
    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if len(xs) < 25:
        return None
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(xs)
        lo, hi = float(np.percentile(xs, 0.5)), float(np.percentile(xs, 99.5))
        if hi <= lo:
            lo, hi = float(xs.min()), float(xs.max())
        pad = 0.05 * (hi - lo + 1e-6)
        grid = np.linspace(lo - pad, hi + pad, n_grid)
        return grid, kde(grid)
    except Exception:
        return None


def _plot_one_marginal(
    ax: plt.Axes,
    sim: np.ndarray,
    obs: np.ndarray,
    *,
    xlabel: str,
    bins: int,
    color_sim: str = "steelblue",
    color_obs: str = "coral",
) -> None:
    sim = np.asarray(sim, dtype=np.float64)
    sim = sim[np.isfinite(sim)]
    obs = np.asarray(obs, dtype=np.float64)
    obs = obs[np.isfinite(obs)]

    ax.hist(sim, bins=bins, density=True, color=color_sim, alpha=0.55, edgecolor="white", label="Simulated")
    if len(obs) > 5:
        ax.hist(obs, bins=bins, density=True, color=color_obs, alpha=0.4, edgecolor="white", label="Observed BIP")

    kde = _kde_x(sim)
    if kde is not None:
        gx, gy = kde
        ax.plot(gx, gy, color="navy", lw=2.0, label="Simulated KDE")
    if len(obs) >= 25:
        kde_o = _kde_x(obs)
        if kde_o is not None:
            gx, gy = kde_o
            ax.plot(gx, gy, color="darkred", lw=2.0, ls="--", label="Observed KDE")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="best")


def plot_marginals_for_hitter(pdir: Path, *, dpi: int = 160) -> list[Path]:
    slug = pdir.name
    sim_path = pdir / "simulated_ev_la_sa_value.parquet"
    base_path = pdir / "baseline_observed_bip.npz"
    if not sim_path.is_file():
        return []

    df = pd.read_parquet(sim_path)
    ev_s = df["EV"].to_numpy()
    la_s = df["LA"].to_numpy()
    sa_s = df["SA"].to_numpy()
    val_s = df["xwobacon"].to_numpy() if "xwobacon" in df.columns else np.array([])

    if base_path.is_file():
        base = np.load(base_path, allow_pickle=True)
        ev_o = np.asarray(base["launch_speed"], dtype=np.float64)
        la_o = np.asarray(base["launch_angle"], dtype=np.float64)
        sa_o = np.asarray(base["spray_angle_deg"], dtype=np.float64)
        val_o = (
            np.asarray(base["xwobacon"], dtype=np.float64)
            if "xwobacon" in base.files
            else np.array([])
        )
    else:
        ev_o = la_o = sa_o = val_o = np.array([])

    display = _title_slug(slug)
    written: list[Path] = []

    panels: list[tuple[str, np.ndarray, np.ndarray, str, int]] = [
        ("ev", ev_s, ev_o, "EV (mph)", 56),
        ("la", la_s, la_o, "LA (deg)", 56),
        ("sa", sa_s, sa_o, "SA (deg)", 56),
    ]
    if len(val_s) > 0:
        panels.append(("value", val_s, val_o, "xwOBAcon (calibrated)", 48))

    for key, sim, obs, xlab, nbins in panels:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        _plot_one_marginal(ax, sim, obs, xlabel=xlab, bins=nbins)
        if key == "value":
            st = f"{display}: value function (xwOBAcon) density (simulated vs observed BIP)"
        else:
            st = f"{display}: {xlab} marginal density (simulated vs observed BIP)"
        fig.suptitle(st, fontsize=12, fontweight="bold")
        out = pdir / f"{slug}_marginal_{key}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print("Wrote", out)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Folder of per-hitter subdirs (default: fullrun). Trunc-ex sims: {DEFAULT_RUN_ROOT_TRUNC_EX}",
    )
    ap.add_argument("--hitter-dir", type=Path, default=None, help="Only this folder (must contain parquet).")
    ap.add_argument("--batch", action="store_true", help="All subdirs of --run-root with simulation parquet.")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    if args.batch:
        root = args.run_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        total = 0
        for p in sorted(d for d in root.iterdir() if d.is_dir()):
            n = len(plot_marginals_for_hitter(p, dpi=args.dpi))
            total += n
        n_players = sum(
            1 for d in root.iterdir() if d.is_dir() and (d / "simulated_ev_la_sa_value.parquet").is_file()
        )
        print(f"Batch done: {n_players} players with parquet, {total} PNGs written under {root}")
        return

    pdir = (args.hitter_dir or (DEFAULT_RUN_ROOT / "aaron_judge")).resolve()
    plot_marginals_for_hitter(pdir, dpi=args.dpi)


if __name__ == "__main__":
    main()
