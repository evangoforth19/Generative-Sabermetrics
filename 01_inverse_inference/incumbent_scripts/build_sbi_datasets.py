#!/usr/bin/env python3
"""Run full SBI data preprocessing (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project package lives under MCMC 2/
_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.dataset_builders import run_full_build  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--production-root",
        type=Path,
        default=_MMC2_ROOT
        / "Outputs"
        / "refactor_exact_root_rhh"
        / "production",
        help="Path to production output folder",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="sbi_forward_sim project root",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--statcast-pickle",
        type=Path,
        default=None,
        help="Pickle with Statcast columns including release_speed (default: MCMC 2/batter_data_2020_2025.pkl)",
    )
    args = p.parse_args()

    out = run_full_build(
        production_root=args.production_root.resolve(),
        project_root=args.project_root.resolve(),
        split_seed=args.split_seed,
        statcast_pickle=args.statcast_pickle,
    )
    print("Done:", out)


if __name__ == "__main__":
    main()
