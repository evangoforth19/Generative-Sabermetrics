#!/usr/bin/env python3
"""Select explicit L1/L2 regularization for reduced-input frozen bounded stage-u MDN."""

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


def _fmt_float_for_path(x: float) -> str:
    return f"{float(x):.0e}".replace("+", "").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "p_u_given_g_reduced_frozen_bounded_reg.yaml",
    )
    ap.add_argument("--lambda-l1", type=float, nargs="+", default=[0.0, 1e-7, 1e-6, 1e-5])
    ap.add_argument("--lambda-l2", type=float, nargs="+", default=[0.0, 1e-6, 1e-5, 1e-4])
    ap.add_argument("--debug-one-epoch", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    sweep_root = project_root / "outputs" / "p_u_given_g_reduced_reg_selection" / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    run_dirs: list[str] = []
    train_script = project_root / "scripts" / "train_p_u_given_g.py"
    candidates: list[tuple[float, float]] = [
        (float(l1), float(l2)) for l1 in args.lambda_l1 for l2 in args.lambda_l2
    ]
    for i, (l1, l2) in enumerate(candidates):
        cfg = copy.deepcopy(base_cfg)
        _set_nested(cfg, "regularization.lambda_l1", l1)
        _set_nested(cfg, "regularization.lambda_l2", l2)
        _set_nested(cfg, "model.n_mixture_components", 10)
        cfg.setdefault("sweep", {})["enabled"] = False
        candidate_name = f"l1_{_fmt_float_for_path(l1)}__l2_{_fmt_float_for_path(l2)}"
        cfg["outputs"]["run_dir"] = (
            f"outputs/p_u_given_g_reduced_reg_selection/{stamp}/candidates/{candidate_name}"
        )
        tmp_cfg = sweep_root / f"candidate_{i:02d}_{candidate_name}.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cmd = [sys.executable, str(train_script), "--config", str(tmp_cfg)]
        if args.debug_one_epoch:
            cmd.append("--debug-one-epoch")
        print(
            f"Training lambda_l1={l1:g} lambda_l2={l2:g} "
            f"({i + 1}/{len(candidates)})",
            flush=True,
        )
        subprocess.run(cmd, check=True, cwd=str(_MMC2_ROOT))

        run_parent = project_root / cfg["outputs"]["run_dir"]
        run_id = sorted(p for p in run_parent.iterdir() if p.is_dir())[-1].name
        run_dir = run_parent / run_id
        run_dirs.append(str(run_dir))
        metrics_path = run_dir / "metrics" / "val_select_pre_temp_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        rows.append(
            {
                "candidate_id": i,
                "lambda_l1": l1,
                "lambda_l2": l2,
                "K": 10,
                "run_dir": str(run_dir),
                "val_select_weighted_nll_pre_temp": metrics.get("weighted_nll"),
                "val_select_weighted_nll_post_temp": manifest.get("nll_val_select_weighted_post_temp"),
                "temp_cal_weighted_nll_post_temp": manifest.get("nll_temp_cal_weighted_post_temp"),
                "test_weighted_nll_post_temp": manifest.get("nll_test_weighted_post_temp"),
            }
        )

    df = pd.DataFrame(rows).sort_values("val_select_weighted_nll_pre_temp", kind="mergesort")
    df.to_csv(sweep_root / "regularization_selection_summary.csv", index=False)
    (sweep_root / "candidate_run_dirs.txt").write_text("\n".join(run_dirs), encoding="utf-8")
    best = df.iloc[0].to_dict()
    report = [
        "# Reduced-input Frozen Bounded Stage-u Regularization Selection",
        "",
        "Selection metric: `val_select_weighted_nll_pre_temp`.",
        "",
        f"Best lambda_l1: `{best['lambda_l1']}`",
        f"Best lambda_l2: `{best['lambda_l2']}`",
        f"Best K: `{int(best['K'])}`",
        f"Best run: `{best['run_dir']}`",
        f"Best validation NLL: `{best['val_select_weighted_nll_pre_temp']}`",
        "",
        "No checkpoint was promoted automatically.",
    ]
    (sweep_root / "model_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Regularization selection complete:", sweep_root, flush=True)


if __name__ == "__main__":
    main()
