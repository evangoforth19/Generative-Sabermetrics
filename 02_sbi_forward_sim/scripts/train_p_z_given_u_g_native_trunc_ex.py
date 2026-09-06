#!/usr/bin/env python3
"""Train stage-z hybrid: 2D Gaussian (x, e_y*) + von Mises(psi) + conditional truncated-normal e_x."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.scripts.train_p_z_given_u_g import _git_hash  # noqa: E402
from sbi_forward_sim.src.data_z import ZArraysNativeTruncEx, load_all_z_data_native_trunc_ex  # noqa: E402
from sbi_forward_sim.src.feature_contract_z import (  # noqa: E402
    load_feature_contract_z,
    resolve_feature_contract_path_z,
)
from sbi_forward_sim.src.models_z import ConditionalHybridGaussVonMisesTruncExZ  # noqa: E402
from sbi_forward_sim.src.train_z import (  # noqa: E402
    _to_tensors_train_trunc_ex,
    _weighted_trunc_ex_joint_nll,
    train_stage_z_hybrid_trunc_ex,
)


def _build_trunc_ex_model(cfg: dict, num_f: int, vocabs: dict[str, dict[str, int]], device: torch.device):
    mcfg = cfg["model"]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesTruncExZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=int(mcfg["n_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        sigma_floor=float(mcfg.get("sigma_floor", 1e-3)),
    ).to(device)


def _eval_weighted_nll_split(
    model: ConditionalHybridGaussVonMisesTruncExZ,
    arr: ZArraysNativeTruncEx,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    x, cat, yg, psi, ex, w = _to_tensors_train_trunc_ex(arr, device)
    n = x.shape[0]
    if n == 0:
        return {"weighted_nll_joint": float("nan"), "mean_nll_joint": float("nan"), "n_rows": 0}
    wsum = 0.0
    wnll = 0.0
    mnll = 0.0
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            wx = w[s:e]
            wn, mn = _weighted_trunc_ex_joint_nll(model, x[s:e], {k: v[s:e] for k, v in cat.items()}, yg[s:e], psi[s:e], ex[s:e], wx)
            wsum += float(wx.sum().item())
            wnll += wn * float(wx.sum().item())
            mnll += mn * float((e - s))
    mean_nll = mnll / max(n, 1)
    weighted = wnll / max(wsum, 1e-12)
    return {"weighted_nll_joint": weighted, "mean_nll_joint": mean_nll, "n_rows": int(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "p_z_given_u_g_native_trunc_ex.yaml",
    )
    ap.add_argument(
        "--resume-eval",
        type=Path,
        default=None,
        help="Existing run directory with checkpoint.pt: skip training, run split NLL metrics only.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume_eval is not None:
        run_dir = args.resume_eval.resolve()
        cfg_path = run_dir / "config_resolved.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        stamp = run_dir.name
        fc_path = run_dir / "feature_contract_frozen.json"
        if not fc_path.is_file():
            raise FileNotFoundError(f"Missing {fc_path}")
        frozen_contract = json.loads(fc_path.read_text(encoding="utf-8"))
        train_a, cal_a, test_a, bundle = load_all_z_data_native_trunc_ex(
            cfg, project_root, feature_contract=frozen_contract
        )
    else:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if cfg.get("target_parameterization") != "native_trunc_ex_vm":
            raise ValueError("This script expects target_parameterization: native_trunc_ex_vm")

        fc = load_feature_contract_z(cfg, project_root)
        for col in fc.get("forbidden_neural_input_columns", []):
            block = (
                list(cfg["upstream_u_features"])
                + list(cfg["numeric_context_features"])
                + list(cfg["player_constant_features"])
            )
            if col in block:
                raise ValueError(f"Config must not place forbidden column {col!r} in neural inputs.")

        train_a, cal_a, test_a, bundle = load_all_z_data_native_trunc_ex(cfg, project_root)
        vocabs = bundle["vocabs"]
        meta = bundle["meta"]

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        run_dir = project_root / cfg["outputs"]["run_dir"] / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

        contract_path = resolve_feature_contract_path_z(cfg, project_root)
        frozen_contract = load_feature_contract_z(cfg, project_root)
        shutil.copy2(contract_path, run_dir / "feature_contract_frozen.json")

        train_stage_z_hybrid_trunc_ex(cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir)

        feature_manifest = {
            "raw_targets_contract": cfg["targets"],
            "target_parameterization": "native_trunc_ex_vm",
            "internal_training_targets_note": "y_gauss std (x,e_y*); psi_rad wrapped; e_x raw [0,0.6] trunc-normal likelihood",
            "upstream_u_features": cfg["upstream_u_features"],
            "numeric_context_features": cfg["numeric_context_features"],
            "player_constant_features": cfg["player_constant_features"],
            "categorical_features": {
                k: {"embedding_dim": v["embedding_dim"], "vocab_size": len(vocabs[k])}
                for k, v in cfg["categorical_features"].items()
            },
            "weight_column": cfg["weight_column"],
            "standardization_stats_path": str(project_root / cfg["paths"]["standardization_stats"]),
            "model_family": cfg["model"].get("family", "hybrid_gauss_vonmises_trunc_ex"),
        }
        (run_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    vocabs = bundle["vocabs"]
    meta = bundle["meta"]

    try:
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)

    num_f = train_a.x_num.shape[1]
    model = _build_trunc_ex_model(cfg, num_f, vocabs, device)
    model.load_state_dict(ckpt["model_state"])

    bs = int(cfg.get("eval", {}).get("eval_splits_batch_size", 4096))
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in [("calibration", cal_a), ("test", test_a)]:
        pack = _eval_weighted_nll_split(model, arr, device, bs)
        (metrics_dir / f"{name}_joint_nll.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")

    run_manifest = {
        "timestamp_utc": stamp,
        "git_hash": _git_hash(),
        "config_path": str((run_dir / "config_resolved.yaml").resolve()),
        "feature_contract_id": frozen_contract.get("contract_id"),
        "target_parameterization": "native_trunc_ex_vm",
        "model_family": "hybrid_gauss_vonmises_trunc_ex",
        "row_counts": {k: int(meta[k]) for k in meta if k.startswith("n_rows")},
        "event_counts": {k: int(meta[k]) for k in meta if k.startswith("n_events")},
        "temperature_T_gauss": float(ckpt.get("temperature_T_gauss", 1.0)),
        "temperature_T_ang": float(ckpt.get("temperature_T_ang", 1.0)),
        "z_targets": meta.get("z_targets", cfg.get("targets")),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print("Done. Outputs:", run_dir)
    print("Calibration / test joint NLL JSON written under metrics/.")


if __name__ == "__main__":
    main()
