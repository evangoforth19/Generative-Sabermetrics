#!/usr/bin/env python3
"""Refit post-hoc z temperatures for native_trunc_ex_vm from an existing checkpoint (weights unchanged).

Same construction as ``recalibrate_p_z_native_circular.py`` for the overlapping blocks:
isotropic or axis-wise Cholesky scaling on the 2D Gaussian (x, e_y*) and κ / T_ang on ψ.
The truncated-normal e_x factor uses the learned μ, σ from the frozen network (not
separately temperature-scaled). See ``fit_hybrid_trunc_ex_temperature`` in
``calibration_z.py``.
"""

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

from sbi_forward_sim.src.calibration_z import fit_hybrid_trunc_ex_temperature  # noqa: E402
from sbi_forward_sim.src.data_z import ZArraysNativeTruncEx, load_all_z_data_native_trunc_ex  # noqa: E402
from sbi_forward_sim.src.models_z import ConditionalHybridGaussVonMisesTruncExZ  # noqa: E402
from sbi_forward_sim.src.train_z import (  # noqa: E402
    _to_tensors_train_trunc_ex,
    _weighted_trunc_ex_joint_nll,
    weighted_trunc_ex_joint_nll_tempered,
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


def _eval_split_joint_nll(
    model: ConditionalHybridGaussVonMisesTruncExZ,
    arr: ZArraysNativeTruncEx,
    device: torch.device,
    batch_size: int,
    *,
    T_gauss_x: float,
    T_gauss_y: float,
    T_ang: float,
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
            xc = {k: v[s:e] for k, v in cat.items()}
            if abs(T_gauss_x - 1.0) < 1e-12 and abs(T_gauss_y - 1.0) < 1e-12 and abs(T_ang - 1.0) < 1e-12:
                wn, mn = _weighted_trunc_ex_joint_nll(model, x[s:e], xc, yg[s:e], psi[s:e], ex[s:e], wx)
            else:
                wn, mn = weighted_trunc_ex_joint_nll_tempered(
                    model,
                    x[s:e],
                    xc,
                    yg[s:e],
                    psi[s:e],
                    ex[s:e],
                    wx,
                    T_gauss_x=T_gauss_x,
                    T_gauss_y=T_gauss_y,
                    T_ang=T_ang,
                )
            wsum += float(wx.sum().item())
            wnll += wn * float(wx.sum().item())
            mnll += mn * float((e - s))
    mean_nll = mnll / max(n, 1)
    weighted = wnll / max(wsum, 1e-12)
    return {"weighted_nll_joint": weighted, "mean_nll_joint": mean_nll, "n_rows": int(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        help="YAML with paths, calibration; defaults to source config_resolved.yaml",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    src = args.source_run_dir.resolve()
    cfg_path = args.config or (src / "config_resolved.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = project_root / cfg["outputs"]["run_dir"] / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "config_resolved.yaml",
        "feature_contract_frozen.json",
        "feature_manifest.json",
        "target_transform_manifest.json",
    ):
        p = src / name
        if p.is_file():
            shutil.copy2(p, run_dir / name)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    try:
        old = torch.load(src / "checkpoint.pt", map_location="cpu", weights_only=False)
    except TypeError:
        old = torch.load(src / "checkpoint.pt", map_location="cpu")

    train_a, cal_a, test_a, bundle = load_all_z_data_native_trunc_ex(cfg, project_root)
    vocabs = bundle["vocabs"]
    device = torch.device("cpu")

    num_f = train_a.x_num.shape[1]
    model = _build_trunc_ex_model(cfg, num_f, vocabs, device)
    model.load_state_dict(old["model_state"])
    model.eval()

    x_cal = torch.from_numpy(cal_a.x_num).to(device)
    cat_cal = {k: torch.from_numpy(v).long().to(device) for k, v in cal_a.cat.items()}
    yg_cal = torch.from_numpy(cal_a.y_gauss).to(device)
    psi_cal = torch.from_numpy(cal_a.psi_rad).to(device)
    ex_cal = torch.from_numpy(cal_a.e_x).to(device)
    w_cal = torch.from_numpy(cal_a.w).to(device)

    with torch.no_grad():
        lg, mg, Lg, mp, kp, h = model(x_cal, cat_cal)
        mu_ex, sig_ex = model.ex_params(h, yg_cal, psi_cal)

    ccal = cfg.get("calibration") or {"max_iter": 400, "lr": 0.05, "anisotropic_gaussian": False}
    htemp = fit_hybrid_trunc_ex_temperature(
        lg.detach(),
        mg.detach(),
        Lg.detach(),
        mp.detach(),
        kp.detach(),
        mu_ex.detach(),
        sig_ex.detach(),
        yg_cal,
        psi_cal,
        ex_cal,
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
        anisotropic_gaussian=bool(ccal.get("anisotropic_gaussian", False)),
    )

    new_ckpt = dict(old)
    new_ckpt["temperature_T_gauss"] = htemp.T_gauss
    new_ckpt["temperature_T_gauss_x"] = htemp.T_gauss_x
    new_ckpt["temperature_T_gauss_y"] = htemp.T_gauss_y
    new_ckpt["temperature_T_ang"] = htemp.T_ang
    new_ckpt["config"] = cfg
    torch.save(new_ckpt, run_dir / "checkpoint.pt")

    temp_note = (
        "native_trunc_ex_vm: Gaussian row-scaled Cholesky (T_gauss geom mean of T_x,T_y); "
        "κ_cal = κ / T_ang for ψ. e_x truncated-normal uses learned σ (unchanged)."
    )
    (run_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_gauss": htemp.T_gauss,
                "T_gauss_x": htemp.T_gauss_x,
                "T_gauss_y": htemp.T_gauss_y,
                "T_ang": htemp.T_ang,
                "note": temp_note,
                "recalibrated_from_run": str(src),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    T_gx_f = float(htemp.T_gauss_x)
    T_gy_f = float(htemp.T_gauss_y)
    T_ang = float(htemp.T_ang)
    bs = int(cfg.get("eval", {}).get("eval_splits_batch_size", 4096))

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for split_name, arr in [("train", train_a), ("calibration", cal_a), ("test", test_a)]:
        pre = _eval_split_joint_nll(model, arr, device, bs, T_gauss_x=1.0, T_gauss_y=1.0, T_ang=1.0)
        post = _eval_split_joint_nll(
            model, arr, device, bs, T_gauss_x=T_gx_f, T_gauss_y=T_gy_f, T_ang=T_ang
        )
        results[f"{split_name}_pre_temp"] = pre
        results[f"{split_name}_post_temp"] = post
        (metrics_dir / f"{split_name}_pre_temp.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
        (metrics_dir / f"{split_name}_post_temp.json").write_text(json.dumps(post, indent=2), encoding="utf-8")

    run_manifest = {
        "timestamp_utc": stamp,
        "config_path": str(cfg_path.resolve()),
        "target_parameterization": "native_trunc_ex_vm",
        "model_family": "hybrid_gauss_vonmises_trunc_ex",
        "temperature_T_gauss": htemp.T_gauss,
        "temperature_T_gauss_x": T_gx_f,
        "temperature_T_gauss_y": T_gy_f,
        "temperature_T_ang": T_ang,
        "recalibrated_from_checkpoint": str(src / "checkpoint.pt"),
        "weighted_nll_joint_test_post_temp": results.get("test_post_temp", {}).get("weighted_nll_joint"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("Done. Outputs:", run_dir)


if __name__ == "__main__":
    main()
