#!/usr/bin/env python3
"""Train stage-z branch_autoreg_bounded_hybrid_mdn_z."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.scripts.train_p_z_given_u_g import _git_hash  # noqa: E402
from sbi_forward_sim.src.calibration_z import BranchAutoregTemperatureScale  # noqa: E402
from sbi_forward_sim.src.data_z_branch_autoreg import load_all_z_data_branch_autoreg_bounded  # noqa: E402
from sbi_forward_sim.src.eval_z_branch_autoreg_bounded import run_rich_eval_all_splits  # noqa: E402
from sbi_forward_sim.src.feature_contract_z import (  # noqa: E402
    load_feature_contract_z,
    resolve_feature_contract_path_z,
)
from sbi_forward_sim.src.models_z import BranchAutoregBoundedHybridMDNZ  # noqa: E402
from sbi_forward_sim.src.split_z_validation import (  # noqa: E402
    save_z_validation_split_metadata,
    split_z_calibration_dataframe,
)
from sbi_forward_sim.src.train_z_branch_autoreg import train_stage_z_branch_autoreg_bounded_mdn  # noqa: E402


def _build_model(cfg: dict, num_f: int, vocabs: dict, device: torch.device) -> BranchAutoregBoundedHybridMDNZ:
    mcfg = cfg["model"]
    cov = cfg.get("covariance", {})
    ex_cfg = cfg.get("ex", {})
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return BranchAutoregBoundedHybridMDNZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=int(mcfg.get("K", 5)),
        hidden_width=int(mcfg.get("hidden_dim", 64)),
        n_hidden=int(mcfg.get("depth", 2)),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(mcfg.get("dropout", 0.0)),
        branch_embedding_dim=int(mcfg.get("branch_embedding_dim", 8)),
        chol_eps=float(cov.get("chol_eps", 1e-4)),
        diag_floor=float(cov.get("diag_floor", 0.05)),
        diag_ceiling=float(cov.get("diag_ceiling", 5.0)),
        use_diag_ceiling=bool(cov.get("use_diag_ceiling", True)),
        offdiag_tanh_scale=float(cov.get("offdiag_tanh_scale", 2.0)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=float(mcfg.get("kappa_max", 120.0)),
        sigma_floor=float(ex_cfg.get("sigma_floor", 0.05)),
        sigma_ceiling=float(ex_cfg.get("sigma_ceiling", 5.0)),
        use_sigma_ceiling=bool(ex_cfg.get("use_sigma_ceiling", True)),
    ).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "p_z_given_u_g_branch_autoreg_bounded_mdn.yaml",
    )
    ap.add_argument("--resume-eval", type=Path, default=None)
    ap.add_argument("--debug-one-epoch", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume_eval is not None:
        run_dir = args.resume_eval.resolve()
        cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        stamp = run_dir.name
        frozen_contract = json.loads((run_dir / "feature_contract_frozen.json").read_text(encoding="utf-8"))
        train_a, val_a, temp_a, test_a, bundle = load_all_z_data_branch_autoreg_bounded(
            cfg, project_root, feature_contract=frozen_contract
        )
        train_meta = {}
    else:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.debug_one_epoch:
            cfg.setdefault("training", {})["debug_one_epoch"] = True
        if cfg.get("model", {}).get("family") != "branch_autoreg_bounded_hybrid_mdn_z":
            raise ValueError("Config model.family must be branch_autoreg_bounded_hybrid_mdn_z")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        run_dir = project_root / cfg["outputs"]["run_dir"] / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

        contract_path = resolve_feature_contract_path_z(cfg, project_root)
        frozen_contract = load_feature_contract_z(cfg, project_root)
        shutil.copy2(contract_path, run_dir / "feature_contract_frozen.json")

        train_a, val_a, temp_a, test_a, bundle = load_all_z_data_branch_autoreg_bounded(cfg, project_root)

        if cfg.get("validation", {}).get("split_calibration_for_model_selection"):
            import pandas as pd

            cal_path = project_root / cfg["paths"]["z_calibration"]
            cal_df = pd.read_parquet(cal_path)
            vfrac = float(cfg["validation"].get("val_select_fraction", 0.5))
            vseed = int(cfg["validation"].get("split_seed", 20260502))
            val_df, temp_df, _ = split_z_calibration_dataframe(cal_df, val_fraction=vfrac, seed=vseed)
            val_ev = set(val_df["event_id"].unique())
            temp_ev = set(temp_df["event_id"].unique())
            save_z_validation_split_metadata(run_dir, val_ev, temp_ev, bundle["meta"].get("validation_split_summary", {}))

        (run_dir / "target_transform.json").write_text(
            json.dumps(bundle["transform_manifest"], indent=2), encoding="utf-8"
        )

        train_meta = train_stage_z_branch_autoreg_bounded_mdn(
            cfg, train_a, val_a, temp_a, bundle["vocabs"], device=device, out_dir=run_dir
        )

    vocabs = bundle["vocabs"]
    meta = bundle["meta"]
    try:
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)

    model = _build_model(cfg, train_a.x_num.shape[1], vocabs, device)
    model.load_state_dict(ckpt["model_state"])
    temp = BranchAutoregTemperatureScale(
        float(ckpt.get("temperature_T_x", 1.0)),
        float(ckpt.get("temperature_T_y", 1.0)),
        float(ckpt.get("temperature_T_psi", 1.0)),
        float(ckpt.get("temperature_T_ex", 1.0)),
    )

    if cfg.get("evaluation", {}).get("rich_eval", True) and not cfg.get("training", {}).get("debug_one_epoch"):
        splits = {"train": train_a, "val_select": val_a, "temp_cal": temp_a, "test": test_a}
        run_rich_eval_all_splits(model, splits, device, run_dir, temp=temp, cfg=cfg)

    report_lines = [
        "# branch_autoreg_bounded_mdn_z report",
        "",
        f"- Run: `{run_dir}`",
        f"- Model family: `branch_autoreg_bounded_hybrid_mdn_z`",
        f"- Best val_select weighted NLL (pre-temp): {train_meta.get('best_val_select_weighted_nll_pre_temp', ckpt.get('best_val_select_weighted_nll_pre_temp'))}",
        f"- Temperatures: T_x={temp.T_x:.4f}, T_y={temp.T_y:.4f}, T_psi={temp.T_psi:.4f}, T_ex={temp.T_ex:.4f}",
        "",
        "See `metrics/*_post_temp.json` and `plots/pit_*_post_temp.png`.",
    ]
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "branch_autoreg_bounded_mdn_z_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": stamp,
                "git_hash": _git_hash(),
                "feature_contract_id": frozen_contract.get("contract_id"),
                "model_family": "branch_autoreg_bounded_hybrid_mdn_z",
                "row_counts": {k: meta[k] for k in meta if k.startswith("n_rows")},
                **{k: v for k, v in train_meta.items() if k != "temperature"},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("Done.", run_dir)


if __name__ == "__main__":
    main()
