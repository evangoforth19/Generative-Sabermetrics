#!/usr/bin/env python3
"""Evaluate a trained p(u|g,p) checkpoint (metrics + PIT plots)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.data_u import load_all_u_data  # noqa: E402
from sbi_forward_sim.src.eval_u import (
    evaluate_hybrid_split_batched,
    evaluate_shared_gaussian_vm_split_batched,
    evaluate_split_batched,
    plot_pit_hist,
    save_metrics_json_csv,
)  # noqa: E402
from sbi_forward_sim.src.feature_contract_u import assert_vocabs_identical  # noqa: E402
from sbi_forward_sim.src.models_u import (  # noqa: E402
    GaussianCholeskyNet,
    HybridBivariateGaussianMixtureDNet,
    SharedMixtureGaussianVonMisesUNet,
    mixture_cdf_1d,
)
from sbi_forward_sim.src.train_u import _arrays_to_tensors  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="e.g. outputs/p_u_given_g/<stamp>")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="If set, reload data with this config; else use checkpoint config",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    device = torch.device("cpu")
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    cfg = ckpt["config"] if args.config is None else yaml.safe_load(args.config.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]

    ckpt_vocabs: dict[str, dict[str, int]] = ckpt["vocabs"]
    train_a, cal_a, test_a, bundle = load_all_u_data(
        cfg, project_root, vocabs_override=ckpt_vocabs
    )
    assert_vocabs_identical(bundle["vocabs"], ckpt_vocabs, context="evaluate_p_u_given_g vocabs")
    vocabs = ckpt_vocabs
    mcfg = cfg["model"]
    num_f = train_a.x_num.shape[1]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    is_hybrid = ckpt.get("head_family") == "hybrid_va_gmm_d"
    is_shared_vm = ckpt.get("head_family") == "shared_gaussian_vm_d"
    if is_shared_vm:
        model = SharedMixtureGaussianVonMisesUNet(
            input_dim=num_f,
            vocab_sizes=vocab_sizes,
            embedding_dims=emb_dims,
            n_mixture=int(mcfg["n_mixture_components"]),
            hidden_width=mcfg["hidden_width"],
            n_hidden=mcfg["hidden_layers"],
            activation=mcfg.get("activation", "tanh"),
            dropout=cfg["training"].get("dropout", 0.0),
            chol_eps=float(mcfg.get("chol_eps", 1e-5)),
            kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
            kappa_max=float(mcfg.get("kappa_max", 120.0)),
        ).to(device)
        T_va = float(ckpt["temperature_T_va"])
        T_ang = float(ckpt["temperature_T_ang"])
    elif is_hybrid:
        model = HybridBivariateGaussianMixtureDNet(
            input_dim=num_f,
            vocab_sizes=vocab_sizes,
            embedding_dims=emb_dims,
            n_mixture=int(mcfg["n_mixture_components"]),
            hidden_width=mcfg["hidden_width"],
            n_hidden=mcfg["hidden_layers"],
            activation=mcfg.get("activation", "tanh"),
            dropout=cfg["training"].get("dropout", 0.0),
            chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        ).to(device)
        T_va = float(ckpt["temperature_T_va"])
        T_mix = float(ckpt["temperature_T_mix"])
    else:
        model = GaussianCholeskyNet(
            input_dim=num_f,
            vocab_sizes=vocab_sizes,
            embedding_dims=emb_dims,
            hidden_width=mcfg["hidden_width"],
            n_hidden=mcfg["hidden_layers"],
            activation=mcfg.get("activation", "tanh"),
            dropout=cfg["training"].get("dropout", 0.0),
            chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        ).to(device)
        T_cal = float(ckpt["temperature_T"])
    model.load_state_dict(ckpt["model_state"])
    tm = torch.tensor(train_a.target_means, dtype=torch.float32, device=device)
    ts = torch.tensor(train_a.target_stds, dtype=torch.float32, device=device)

    out_m = run_dir / "metrics_reval"
    for name, arr in [("train", train_a), ("calibration", cal_a), ("test", test_a)]:
        x, cat, y, yr, w = _arrays_to_tensors(arr, device)
        if is_shared_vm:
            for label, tva, tang in [
                (f"{name}_pre_temp", 1.0, 1.0),
                (f"{name}_post_temp", T_va, T_ang),
            ]:
                pack = evaluate_shared_gaussian_vm_split_batched(
                    model,
                    x,
                    cat,
                    y,
                    yr,
                    w,
                    tm,
                    ts,
                    T_va=tva,
                    T_ang=tang,
                    batch_size=4096,
                    device=device,
                    kappa_max=float(ckpt.get("kappa_max", 120.0)),
                )
                save_metrics_json_csv(out_m, label, pack)
            pit_pack = evaluate_shared_gaussian_vm_split_batched(
                model,
                x,
                cat,
                y,
                yr,
                w,
                tm,
                ts,
                T_va=T_va,
                T_ang=T_ang,
                batch_size=4096,
                device=device,
                kappa_max=float(ckpt.get("kappa_max", 120.0)),
                return_pit_array=True,
            )
            pva = pit_pack.get("pit_values")
            if pva is not None:
                plot_pit_hist(pva, run_dir / "plots_reval" / f"pit_{name}_post_temp.png")
        elif is_hybrid:
            for label, tva, tmix in [
                (f"{name}_pre_temp", 1.0, 1.0),
                (f"{name}_post_temp", T_va, T_mix),
            ]:
                pack = evaluate_hybrid_split_batched(
                    model,
                    x,
                    cat,
                    y,
                    yr,
                    w,
                    tm,
                    ts,
                    T_va=tva,
                    T_mix=tmix,
                    batch_size=8192,
                    device=device,
                )
                save_metrics_json_csv(out_m, label, pack)
            pits = []
            bs = 8192
            with torch.no_grad():
                for s in range(0, x.shape[0], bs):
                    e = min(s + bs, x.shape[0])
                    mva, Lva, mlog, mm, ms = model(x[s:e], {k: v[s:e] for k, v in cat.items()})
                    Ls = Lva * float(np.sqrt(T_va))
                    sig_mix = ms * float(np.sqrt(T_mix))
                    Sig2 = torch.bmm(Ls, Ls.transpose(-1, -2))
                    var_va = torch.diagonal(Sig2, dim1=-2, dim2=-1).clamp_min(1e-12)
                    std_va = torch.sqrt(var_va)
                    inv0 = 1.0 / std_va[:, 0].clamp_min(1e-8)
                    inv1 = 1.0 / std_va[:, 1].clamp_min(1e-8)
                    p0 = torch.special.ndtr((y[s:e, 0] - mva[:, 0]) * inv0)
                    p1 = torch.special.ndtr((y[s:e, 1] - mva[:, 1]) * inv1)
                    p2 = mixture_cdf_1d(y[s:e, 2], mlog, mm, sig_mix).clamp(0.0, 1.0)
                    pits.append(torch.stack([p0, p1, p2], dim=-1).detach().cpu().numpy())
            plot_pit_hist(np.concatenate(pits, axis=0), run_dir / "plots_reval" / f"pit_{name}_post_temp.png")
        else:
            for label, T_use in [(f"{name}_pre_temp", 1.0), (f"{name}_post_temp", T_cal)]:
                pack = evaluate_split_batched(
                    model, x, cat, y, yr, w, tm, ts, T=T_use, batch_size=8192, device=device
                )
                save_metrics_json_csv(out_m, label, pack)
            pits = []
            bs = 8192
            with torch.no_grad():
                for s in range(0, x.shape[0], bs):
                    e = min(s + bs, x.shape[0])
                    mu, L = model(x[s:e], {k: v[s:e] for k, v in cat.items()})
                    Ls = L * float(np.sqrt(T_cal))
                    var_m = torch.diagonal(
                        torch.bmm(Ls, Ls.transpose(-1, -2)), dim1=-2, dim2=-1
                    ).clamp_min(1e-12)
                    std_m = torch.sqrt(var_m)
                    z = (y[s:e] - mu) / std_m.clamp_min(1e-8)
                    pits.append(torch.special.ndtr(z).detach().cpu().numpy())
            plot_pit_hist(np.concatenate(pits, axis=0), run_dir / "plots_reval" / f"pit_{name}_post_temp.png")

    print("Wrote metrics to", out_m)


if __name__ == "__main__":
    main()
