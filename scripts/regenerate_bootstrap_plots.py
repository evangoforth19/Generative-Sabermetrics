#!/usr/bin/env python3
"""Regenerate overlay + marginal PNGs for a bootstrap run directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=REPO / "outputs/sbi_bootstrap_value_per_hitter_trunc_ex_vm_calibrated_20260508",
    )
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    root = args.bootstrap_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    py = sys.executable
    subprocess.run(
        [py, str(REPO / "scripts/overlay_bootstrap_empirical_vs_simulated.py"), "--bootstrap-dir", str(root)],
        check=True,
    )
    subprocess.run(
        [
            py,
            str(REPO / "scripts/plot_hitter_marginal_densities.py"),
            "--run-root",
            str(root),
            "--batch",
            "--dpi",
            str(args.dpi),
        ],
        check=True,
    )
    print("Regenerated overlays and marginal plots under", root)


if __name__ == "__main__":
    main()
