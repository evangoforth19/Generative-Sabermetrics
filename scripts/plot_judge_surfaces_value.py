#!/usr/bin/env python3
"""
Plot EV/LA/SA joint surfaces and xwOBAcon value distribution from SBI bootstrap outputs.

Single hitter (default legacy path) or batch all subfolders under ``--run-root``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO / "outputs/sbi_bootstrap_value_per_hitter_fullrun"
DEFAULT_SINGLE = DEFAULT_RUN_ROOT / "aaron_judge"


def _title_from_slug(slug: str) -> str:
    """e.g. aaron_judge -> Aaron Judge"""
    return re.sub(r"\s+", " ", slug.replace("_", " ").strip()).title()


def plot_one_hitter(pdir: Path, *, dpi: int = 160) -> Path | None:
    """Write ``{slug}_ev_la_sa_surfaces_and_value.png`` under ``pdir``. Returns path or None if skipped."""
    sim = pdir / "simulated_ev_la_sa_value.parquet"
    base_npz = pdir / "baseline_observed_bip.npz"
    if not sim.is_file():
        print("skip (no simulated_ev_la_sa_value.parquet):", pdir)
        return None

    slug = pdir.name
    out_png = pdir / f"{slug}_ev_la_sa_surfaces_and_value.png"
    display = _title_from_slug(slug)

    df = pd.read_parquet(sim)
    if len(df) < 5:
        print("skip (too few rows):", pdir, "n=", len(df))
        return None

    ev = df["EV"].to_numpy()
    la = df["LA"].to_numpy()
    sa = df["SA"].to_numpy()
    v = df["xwobacon"].to_numpy()

    base = np.load(base_npz, allow_pickle=True) if base_npz.is_file() else None
    if base is not None:
        ev_o = base["launch_speed"]
        la_o = base["launch_angle"]
        sa_o = base["spray_angle_deg"]
    else:
        ev_o = la_o = sa_o = np.array([])

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    ax_el = fig.add_subplot(gs[0, 0])
    ax_es = fig.add_subplot(gs[0, 1])
    ax_ls = fig.add_subplot(gs[1, 0])
    ax_v = fig.add_subplot(gs[1, 1])

    bins_el = (55, 48)
    h_el, xe, ye = np.histogram2d(ev, la, bins=bins_el, range=[[40, 125], [-60, 60]])
    h_el = np.maximum(h_el, 1e-12)
    im0 = ax_el.pcolormesh(
        xe, ye, h_el.T, shading="auto", cmap="viridis", norm=LogNorm(vmin=h_el[h_el > 0].min(), vmax=h_el.max())
    )
    if len(ev_o) > 0:
        ax_el.scatter(ev_o, la_o, s=2, c="white", alpha=0.12, linewidths=0, rasterized=True, label="Observed BIP (faint)")
    ax_el.set_xlabel("EV (mph)")
    ax_el.set_ylabel("LA (deg)")
    ax_el.set_title("Simulated density: EV vs LA\n(SBI bootstrap → decoder)")
    fig.colorbar(im0, ax=ax_el, label="count (log scale)")
    if len(ev_o) > 0:
        ax_el.legend(loc="upper right", fontsize=8)

    bins_es = (55, 40)
    h_es, xe2, ye2 = np.histogram2d(ev, sa, bins=bins_es, range=[[40, 125], [-50, 50]])
    h_es = np.maximum(h_es, 1e-12)
    im1 = ax_es.pcolormesh(
        xe2, ye2, h_es.T, shading="auto", cmap="magma", norm=LogNorm(vmin=h_es[h_es > 0].min(), vmax=h_es.max())
    )
    if len(ev_o) > 0:
        ax_es.scatter(ev_o, sa_o, s=2, c="cyan", alpha=0.1, linewidths=0, rasterized=True)
    ax_es.set_xlabel("EV (mph)")
    ax_es.set_ylabel("SA (deg)")
    ax_es.set_title("Simulated density: EV vs SA")
    fig.colorbar(im1, ax=ax_es, label="count (log scale)")

    bins_ls = (48, 40)
    h_ls, xe3, ye3 = np.histogram2d(la, sa, bins=bins_ls, range=[[-60, 60], [-50, 50]])
    h_ls = np.maximum(h_ls, 1e-12)
    im2 = ax_ls.pcolormesh(
        xe3, ye3, h_ls.T, shading="auto", cmap="cividis", norm=LogNorm(vmin=h_ls[h_ls > 0].min(), vmax=h_ls.max())
    )
    if len(ev_o) > 0:
        ax_ls.scatter(la_o, sa_o, s=2, c="orange", alpha=0.1, linewidths=0, rasterized=True)
    ax_ls.set_xlabel("LA (deg)")
    ax_ls.set_ylabel("SA (deg)")
    ax_ls.set_title("Simulated density: LA vs SA")
    fig.colorbar(im2, ax=ax_ls, label="count (log scale)")

    ax_v.hist(v, bins=56, color="steelblue", edgecolor="white", alpha=0.85, density=True, label="Simulated xwOBAcon")
    if base is not None and "xwobacon" in base.files:
        vo = np.asarray(base["xwobacon"], dtype=np.float64)
        vo = vo[np.isfinite(vo)]
        if len(vo) > 10:
            ax_v.hist(vo, bins=56, color="coral", edgecolor="white", alpha=0.45, density=True, label="Observed BIP xwOBAcon")
    ax_v.set_xlabel("xwOBAcon (LGBM surrogate)")
    ax_v.set_ylabel("density")
    ax_v.set_title("Value function distribution")
    ax_v.legend(fontsize=9)

    fig.suptitle(f"{display} — batted-ball surrogate over SBI/decoded (EV, LA, SA)", fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("Wrote", out_png)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Batch mode: parent directory containing one folder per hitter.",
    )
    ap.add_argument(
        "--hitter-dir",
        type=Path,
        default=None,
        help="If set, only plot this directory (e.g. .../aaron_judge).",
    )
    ap.add_argument("--batch", action="store_true", help="Plot every subdir of --run-root that has simulation parquet.")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    if args.batch:
        root = args.run_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        n = 0
        for p in subdirs:
            if plot_one_hitter(p, dpi=args.dpi):
                n += 1
        print(f"Batch done: {n} figure(s) under {root}")
        return

    pdir = (args.hitter_dir or DEFAULT_SINGLE).resolve()
    plot_one_hitter(pdir, dpi=args.dpi)


if __name__ == "__main__":
    main()
