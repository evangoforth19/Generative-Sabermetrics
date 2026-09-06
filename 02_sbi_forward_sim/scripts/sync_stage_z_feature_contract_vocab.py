#!/usr/bin/env python3
"""
After rebuilding z_* from basic-screened draws, refresh stage-z feature contract vocab sizes
from z_train (train-only categories, same rule as training).

Writes configs/feature_contracts/p_z_given_u_g_feature_contract_v2.json
and prints a one-line reminder to point p_z_given_u_g_native_circular.yaml paths.feature_contract at it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

import pandas as pd  # noqa: E402

from sbi_forward_sim.src.data_u import build_categorical_vocabs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "feature_contracts"
        / "p_z_given_u_g_feature_contract.json",
    )
    ap.add_argument(
        "--z-train",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_processed" / "z_train.parquet",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "feature_contracts"
        / "p_z_given_u_g_feature_contract_v2.json",
    )
    args = ap.parse_args()

    base = json.loads(args.base_contract.read_text(encoding="utf-8"))
    df = pd.read_parquet(args.z_train)
    cat_order = list(base["categorical_yaml_key_order"])
    vocabs = build_categorical_vocabs(df, cat_order)
    for name in cat_order:
        base["categorical_features"][name]["vocab_size"] = len(vocabs[name])
    base["contract_id"] = "p_z_given_u_g_stage_z_v2"
    base["description"] = (
        "Stage-z v2: same neural semantics as v1; vocab sizes synced to basic-screened z_train.parquet "
        f"({len(df):,} rows). Frozen u/g/p blocks unchanged."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    # Preserve a dated backup of v1 next to v2 once (optional copy if user wants)
    print("Wrote", args.out)
    print("Vocab sizes:", {k: len(vocabs[k]) for k in cat_order})
    root = Path(__file__).resolve().parents[1]
    try:
        rel = args.out.relative_to(root)
    except ValueError:
        rel = args.out
    print("Set paths.feature_contract to:", str(rel))


if __name__ == "__main__":
    main()
