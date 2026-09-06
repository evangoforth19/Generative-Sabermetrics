#!/usr/bin/env python3
"""Hyperparameter sweep for bounded_player_support_hier stage-u (rank on val_select)."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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


def _grid_candidates(base: dict) -> list[dict]:
    sweep = base.get("sweep", {})
    grid = sweep.get("grid", {})
    if not grid:
        return [base]
    keys = list(grid.keys())
    vals = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        c = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _set_nested(c, k, v)
        out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--final-test-selected-only", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sweep_root = project_root / "outputs" / "p_u_given_g_sweeps" / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    candidates = _grid_candidates(base_cfg)
    if args.max_runs:
        candidates = candidates[: args.max_runs]

    rows: list[dict] = []
    run_dirs: list[str] = []
    train_script = project_root / "scripts" / "train_p_u_given_g.py"

    for i, cfg in enumerate(candidates):
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("sweep", {})["enabled"] = False
        cfg["training"]["max_epochs"] = min(int(cfg["training"].get("max_epochs", 120)), 30)
        tmp_cfg = sweep_root / f"candidate_{i:03d}.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        print(f"Training candidate {i+1}/{len(candidates)} -> {tmp_cfg}")
        subprocess.run(
            [sys.executable, str(train_script), "--config", str(tmp_cfg)],
            check=True,
            cwd=str(_MMC2_ROOT),
        )
        run_id = sorted((project_root / cfg["outputs"]["run_dir"]).iterdir())[-1].name
        run_dir = project_root / cfg["outputs"]["run_dir"] / run_id
        run_dirs.append(str(run_dir))
        metrics_path = run_dir / "metrics" / "val_select_post_temp_metrics.json"
        if not metrics_path.is_file():
            metrics_path = run_dir / "metrics" / "val_select_pre_temp_metrics.json"
        m = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        rows.append(
            {
                "candidate_id": i,
                "run_dir": str(run_dir),
                "val_select_weighted_nll": m.get("weighted_nll"),
                "model_K": cfg["model"].get("n_mixture_components"),
                "player_embedding_dim": cfg["model"].get("player_embedding_dim"),
                "lambda_player_embedding_l2": cfg.get("regularization", {}).get(
                    "lambda_player_embedding_l2"
                ),
                "diag_floor": cfg.get("covariance", {}).get("diag_floor"),
                "lr": cfg["training"].get("lr"),
            }
        )

    df = pd.DataFrame(rows).sort_values("val_select_weighted_nll")
    df.to_csv(sweep_root / "sweep_summary.csv", index=False)
    (sweep_root / "candidate_run_dirs.txt").write_text("\n".join(run_dirs), encoding="utf-8")
    (sweep_root / "sweep_config.json").write_text(
        json.dumps({"base_config": str(args.base_config), "n_candidates": len(candidates)}, indent=2),
        encoding="utf-8",
    )
    best = df.iloc[0]
    report = [
        "# Sweep selection report",
        "",
        f"Best candidate: `{best['run_dir']}`",
        f"val_select weighted NLL: **{best['val_select_weighted_nll']}**",
        "",
        "Not promoted as canonical automatically.",
    ]
    (sweep_root / "final_model_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Sweep done:", sweep_root)


if __name__ == "__main__":
    main()
