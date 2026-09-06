#!/usr/bin/env python3
"""Hyperparameter sweep for branch_autoreg_bounded_hybrid_mdn_z (val_select only; test once at end)."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))


def _set_nested(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _grid(sweep_cfg: dict) -> list[dict]:
    grid = sweep_cfg.get("grid", {})
    keys = sorted(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        out.append(dict(zip(keys, combo)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--final-test-selected-only", action="store_true", default=True)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = all grid points")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    sweep = base.get("sweep", {})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sweep_root = project_root / "outputs" / "p_z_given_u_g_sweeps" / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    combos = _grid(sweep)
    if args.max_candidates > 0:
        combos = combos[: args.max_candidates]

    train_script = project_root / "scripts" / "train_p_z_given_u_g_branch_autoreg_bounded_mdn.py"
    rows: list[dict] = []
    run_dirs: list[str] = []

    for i, combo in enumerate(combos):
        cfg = copy.deepcopy(base)
        for k, v in combo.items():
            _set_nested(cfg, k, v)
        cfg["training"]["debug_one_epoch"] = True  # sweep uses 1-epoch smoke per candidate unless overridden
        if not sweep.get("use_debug_one_epoch", True):
            cfg["training"]["debug_one_epoch"] = False
        cand_dir = sweep_root / f"candidate_{i:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cand_dir / "candidate_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        cmd = [sys.executable, str(train_script), "--config", str(cfg_path)]
        print("Running", cmd)
        subprocess.run(cmd, cwd=str(_MMC2_ROOT), check=True)
        # train script writes to outputs/p_z_given_u_g/<stamp> — find latest
        out_base = project_root / cfg["outputs"]["run_dir"]
        latest = sorted(out_base.iterdir())[-1]
        run_dirs.append(str(latest))
        manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
        val_nll = manifest.get("best_val_select_weighted_nll_pre_temp", float("inf"))
        row = {"candidate": i, "run_dir": str(latest), "val_select_nll": val_nll, **combo}
        rows.append(row)

    rows.sort(key=lambda r: r["val_select_nll"])
    csv_path = sweep_root / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["candidate"])
        w.writeheader()
        w.writerows(rows)

    (sweep_root / "sweep_config.json").write_text(json.dumps({"base": str(args.base_config), "n_candidates": len(rows)}, indent=2))
    (sweep_root / "candidate_run_dirs.txt").write_text("\n".join(run_dirs), encoding="utf-8")

    best = rows[0] if rows else None
    report = [
        "# Sweep final selection",
        "",
        f"Best candidate: `{best['run_dir'] if best else 'n/a'}`",
        f"val_select NLL: {best['val_select_nll'] if best else 'n/a'}",
        "",
        "Re-train best config without debug_one_epoch, then evaluate on z_test.",
    ]
    (sweep_root / "final_model_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Sweep done.", sweep_root)


if __name__ == "__main__":
    main()
