#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def _safe_name(p: Path) -> str:
    return p.name.replace("_", " ").title()


def _load_arrays(hitter_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sim = pd.read_parquet(hitter_dir / "simulated_ev_la_sa_value.parquet")
    obs = np.load(hitter_dir / "baseline_observed_bip.npz")

    ev_sim = pd.to_numeric(sim["EV"], errors="coerce").to_numpy(dtype=np.float64)
    la_sim = pd.to_numeric(sim["LA"], errors="coerce").to_numpy(dtype=np.float64)
    sa_sim = pd.to_numeric(sim["SA"], errors="coerce").to_numpy(dtype=np.float64)
    xv_sim = pd.to_numeric(sim["xwobacon"], errors="coerce").to_numpy(dtype=np.float64)

    ev_obs = np.asarray(obs["launch_speed"], dtype=np.float64)
    la_obs = np.asarray(obs["launch_angle"], dtype=np.float64)
    sa_obs = np.asarray(obs["spray_angle_deg"], dtype=np.float64)
    xv_obs = np.asarray(obs["xwobacon"], dtype=np.float64)
    return ev_sim, la_sim, sa_sim, xv_sim, ev_obs, la_obs, sa_obs, xv_obs


def _plot_overlay(hitter_dir: Path) -> Path:
    ev_sim, la_sim, sa_sim, xv_sim, ev_obs, la_obs, sa_obs, xv_obs = _load_arrays(hitter_dir)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("EV (mph)", ev_obs, ev_sim, axes[0, 0], "steelblue"),
        ("LA (deg)", la_obs, la_sim, axes[0, 1], "darkorange"),
        ("SA (deg)", sa_obs, sa_sim, axes[1, 0], "seagreen"),
        ("xwOBAcon", xv_obs, xv_sim, axes[1, 1], "purple"),
    ]
    for title, obs, sim, ax, color in specs:
        obs = obs[np.isfinite(obs)]
        sim = sim[np.isfinite(sim)]
        if obs.size == 0 or sim.size == 0:
            ax.set_title(f"{title} (missing)")
            continue
        allv = np.concatenate([obs, sim])
        bins = min(72, max(24, int(np.sqrt(float(allv.size)))))
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(obs, bins=edges, density=True, alpha=0.45, color="gray", label=f"empirical (n={obs.size})")
        ax.hist(sim, bins=edges, density=True, alpha=0.45, color=color, label=f"simulated (n={sim.size})")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(_safe_name(hitter_dir), fontsize=12)
    fig.tight_layout()
    out = hitter_dir / "overlay_empirical_vs_simulated_ev_la_sa_value.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=Path("outputs/sbi_bootstrap_value_per_hitter_trunc_ex_vm_calibrated_20260508"),
    )
    args = ap.parse_args()

    root = args.bootstrap_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    hitter_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    created: list[Path] = []
    for d in hitter_dirs:
        if not (d / "simulated_ev_la_sa_value.parquet").is_file():
            continue
        if not (d / "baseline_observed_bip.npz").is_file():
            continue
        created.append(_plot_overlay(d))

    print(f"Created {len(created)} overlay plot(s).")
    for p in created:
        print(str(p))


if __name__ == "__main__":
    main()

