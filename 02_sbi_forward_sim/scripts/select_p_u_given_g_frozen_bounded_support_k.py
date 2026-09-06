#!/usr/bin/env python3
"""Select K for frozen bounded-support stage-u MDN on val_select."""

from __future__ import annotations

import argparse
import copy
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
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "p_u_given_g_frozen_bounded_support.yaml",
    )
    ap.add_argument("--k", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--debug-one-epoch", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sweep_root = project_root / "outputs" / "p_u_given_g_k_selection" / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    run_dirs: list[str] = []
    train_script = project_root / "scripts" / "train_p_u_given_g.py"
    for i, k in enumerate(args.k):
        cfg = copy.deepcopy(base_cfg)
        _set_nested(cfg, "model.n_mixture_components", int(k))
        cfg.setdefault("sweep", {})["enabled"] = False
        tmp_cfg = sweep_root / f"candidate_K{k}.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cmd = [sys.executable, str(train_script), "--config", str(tmp_cfg)]
        if args.debug_one_epoch:
            cmd.append("--debug-one-epoch")
        print(f"Training K={k} ({i + 1}/{len(args.k)})")
        subprocess.run(cmd, check=True, cwd=str(_MMC2_ROOT))

        run_id = sorted((project_root / cfg["outputs"]["run_dir"]).iterdir())[-1].name
        run_dir = project_root / cfg["outputs"]["run_dir"] / run_id
        run_dirs.append(str(run_dir))
        metrics_path = run_dir / "metrics" / "val_select_pre_temp_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        rows.append(
            {
                "candidate_id": i,
                "K": int(k),
                "run_dir": str(run_dir),
                "val_select_weighted_nll_pre_temp": metrics.get("weighted_nll"),
                "temp_cal_weighted_nll_post_temp": manifest.get("nll_temp_cal_weighted_post_temp"),
                "test_weighted_nll_post_temp": manifest.get("nll_test_weighted_post_temp"),
            }
        )

    df = pd.DataFrame(rows).sort_values("val_select_weighted_nll_pre_temp")
    df.to_csv(sweep_root / "k_selection_summary.csv", index=False)
    (sweep_root / "candidate_run_dirs.txt").write_text("\n".join(run_dirs), encoding="utf-8")
    best = df.iloc[0].to_dict()
    report = [
        "# Frozen Bounded-support Stage-u K Selection",
        "",
        "Selection metric: `val_select_weighted_nll_pre_temp`.",
        "",
        f"Best K: `{int(best['K'])}`",
        f"Best run: `{best['run_dir']}`",
        f"Best validation NLL: `{best['val_select_weighted_nll_pre_temp']}`",
        "",
        "No checkpoint was promoted automatically.",
    ]
    (sweep_root / "model_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    print("K-selection complete:", sweep_root)


if __name__ == "__main__":
    main()
