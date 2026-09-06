#!/usr/bin/env python3
"""Train stage-u conditional Gaussian model p(u | g, p)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.data_u import (  # noqa: E402
    load_all_u_data,
    load_all_u_data_player_support,
    uses_bounded_va_transform,
)
from sbi_forward_sim.src.target_transforms_u import uses_player_specific_bounded_transform  # noqa: E402
from sbi_forward_sim.src.eval_u import (
    evaluate_bounded_hier_shared_gaussian_vm_split_batched,
    evaluate_hybrid_split_batched,
    evaluate_split_batched,
    plot_pit_hist,
    save_metrics_json_csv,
    save_player_embedding_norms_csv,
)  # noqa: E402
from sbi_forward_sim.src.feature_contract_u import load_feature_contract, resolve_feature_contract_path  # noqa: E402
from sbi_forward_sim.src.models_u import (  # noqa: E402
    BoundedHierSharedMixtureGaussianVonMisesUNet,
    GaussianCholeskyNet,
    HybridBivariateGaussianMixtureDNet,
    SharedMixtureGaussianVonMisesUNet,
    mixture_cdf_1d,
)
from sbi_forward_sim.src.train_u import (  # noqa: E402
    FROZEN_BOUNDED_SUPPORT_HEAD,
    _arrays_to_tensors,
    _arrays_to_tensors_bounded,
    _n_mixture_from_cfg,
    build_bounded_player_support_model,
    train_stage_u,
    train_stage_u_frozen_bounded_support,
    train_stage_u_bounded_hier_shared_gaussian_vonmises,
    train_stage_u_bounded_player_support,
    train_stage_u_hybrid_gmm_d,
    train_stage_u_shared_gaussian_vonmises,
)
from sbi_forward_sim.src.target_transforms_u import (  # noqa: E402
    BoundedLogitZScoreTransform,
    PlayerSpecificBoundedLogitZScoreTransform,
    uses_player_specific_bounded_transform,
)
from sbi_forward_sim.src.eval_u import evaluate_shared_gaussian_vm_split_batched  # noqa: E402


def _is_hybrid_va_gmm_d(cfg: dict) -> bool:
    return cfg.get("model", {}).get("head_family") == "hybrid_va_gmm_d"


def _is_shared_gaussian_vm_d(cfg: dict) -> bool:
    return cfg.get("model", {}).get("head_family") == "shared_gaussian_vm_d"


def _is_frozen_bounded_support(cfg: dict) -> bool:
    return cfg.get("model", {}).get("head_family") == FROZEN_BOUNDED_SUPPORT_HEAD


def _is_bounded_hier_shared_gaussian_vm_d(cfg: dict) -> bool:
    return cfg.get("model", {}).get("head_family") == "bounded_hier_shared_gaussian_vm_d"


def _is_bounded_player_support(cfg: dict) -> bool:
    return (
        cfg.get("model", {}).get("head_family")
        == "bounded_player_support_hier_shared_gaussian_vm_d"
    )


def _build_model(
    cfg: dict, num_f: int, vocabs: dict[str, dict[str, int]], device: torch.device
) -> torch.nn.Module:
    mcfg = cfg["model"]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    if _is_bounded_player_support(cfg):
        return build_bounded_player_support_model(cfg, num_f, vocabs, device)
    if _is_bounded_hier_shared_gaussian_vm_d(cfg):
        cvcfg = cfg.get("covariance", {})
        mcfg = cfg["model"]
        vocab_sizes_ord = {k: len(vocabs[k]) for k in ("pitch_type", "stand", "p_throws")}
        return BoundedHierSharedMixtureGaussianVonMisesUNet(
            input_dim=num_f,
            n_batters=len(vocabs["batter_name"]),
            vocab_sizes=vocab_sizes_ord,
            embedding_dims=emb_dims,
            n_mixture=_n_mixture_from_cfg(mcfg),
            hidden_width=int(mcfg.get("hidden_width", 64)),
            n_hidden=int(mcfg.get("hidden_layers", 2)),
            activation=mcfg.get("activation", "tanh"),
            dropout=cfg["training"].get("dropout", 0.0),
            player_embedding_dim=int(mcfg.get("player_embedding_dim", 8)),
            player_effect_scale=float(mcfg.get("player_effect_scale", 1.0)),
            chol_eps=float(cvcfg.get("chol_eps", 1e-4)),
            diag_floor=float(cvcfg.get("diag_floor", 0.05)),
            diag_ceiling=float(cvcfg.get("diag_ceiling", 5.0)),
            use_diag_ceiling=bool(cvcfg.get("use_diag_ceiling", True)),
            kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
            kappa_max=float(mcfg.get("kappa_max", 120.0)),
        ).to(device)
    if _is_shared_gaussian_vm_d(cfg) or _is_frozen_bounded_support(cfg):
        return SharedMixtureGaussianVonMisesUNet(
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
    if _is_hybrid_va_gmm_d(cfg):
        return HybridBivariateGaussianMixtureDNet(
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
    return GaussianCholeskyNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        hidden_width=mcfg["hidden_width"],
        n_hidden=mcfg["hidden_layers"],
        activation=mcfg.get("activation", "tanh"),
        dropout=cfg["training"].get("dropout", 0.0),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
    ).to(device)


def _git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_MMC2_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "p_u_given_g_default.yaml",
    )
    ap.add_argument(
        "--debug-one-epoch",
        action="store_true",
        help="Train exactly one epoch (smoke / debug).",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = project_root / cfg["outputs"]["run_dir"] / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    contract_path = resolve_feature_contract_path(cfg, project_root)
    frozen_contract = load_feature_contract(cfg, project_root)
    shutil.copy2(contract_path, run_dir / "feature_contract_frozen.json")

    cal_a = None
    val_a = temp_cal_a = None
    if _is_bounded_player_support(cfg) or _is_frozen_bounded_support(cfg):
        bundle_ps = load_all_u_data_player_support(
            cfg, project_root, run_dir=run_dir
        )
        train_a = bundle_ps["train"]
        val_a = bundle_ps["val_select"]
        temp_cal_a = bundle_ps["temp_cal"]
        test_a = bundle_ps["test"]
        bundle = bundle_ps
        vocabs = bundle["vocabs"]
        meta = bundle["meta"]
        va_transform = bundle["va_transform"]
        player_bounds = bundle["player_bounds"]
    else:
        train_a, cal_a, test_a, bundle = load_all_u_data(cfg, project_root)
        vocabs = bundle["vocabs"]
        meta = bundle["meta"]
        va_transform = bundle.get("va_transform")
        player_bounds = None
        if meta.get("bounds_diagnostics"):
            (run_dir / "bounds_diagnostics.json").write_text(
                json.dumps(meta["bounds_diagnostics"], indent=2), encoding="utf-8"
            )

    feature_manifest = {
        "targets": cfg["targets"],
        "numeric_features": cfg["numeric_features"],
        "player_constant_features": cfg["player_constant_features"],
        "categorical_features": {
            k: {"embedding_dim": v["embedding_dim"], "vocab_size": len(vocabs[k])}
            for k, v in cfg["categorical_features"].items()
        },
        "weight_column": cfg["weight_column"],
        "standardization_stats_path": str(project_root / cfg["paths"]["standardization_stats"]),
        "player_constant_standardization": (
            "train-only mean/std computed at load time (not in preprocessing JSON): "
            f"{meta.get('player_constant_stats_from_train_only', [])}"
        ),
        "head_family": cfg.get("model", {}).get("head_family", "full_gaussian_cholesky"),
    }
    if (
        _is_hybrid_va_gmm_d(cfg)
        or _is_shared_gaussian_vm_d(cfg)
        or _is_frozen_bounded_support(cfg)
        or _is_bounded_hier_shared_gaussian_vm_d(cfg)
        or _is_bounded_player_support(cfg)
    ):
        feature_manifest["n_mixture_components"] = _n_mixture_from_cfg(cfg["model"])
    if uses_player_specific_bounded_transform(cfg):
        feature_manifest["target_transform"] = "player_specific_bounded_logit_zscore"
    elif uses_bounded_va_transform(cfg):
        feature_manifest["target_transform"] = "bounded_logit_zscore"
    (run_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    device = torch.device("cpu")
    t0 = time.time()
    if _is_frozen_bounded_support(cfg):
        if not isinstance(va_transform, PlayerSpecificBoundedLogitZScoreTransform):
            raise ValueError("frozen bounded support model requires player-specific bounded transform")
        train_meta = train_stage_u_frozen_bounded_support(
            cfg,
            train_a,
            val_a,
            temp_cal_a,
            vocabs,
            device=device,
            out_dir=run_dir,
            va_transform=va_transform,
            player_bounds=player_bounds,
            debug_one_epoch=args.debug_one_epoch,
        )
    elif _is_bounded_player_support(cfg):
        train_meta = train_stage_u_bounded_player_support(
            cfg,
            train_a,
            val_a,
            temp_cal_a,
            vocabs,
            device=device,
            out_dir=run_dir,
            va_transform=va_transform,
            player_bounds=player_bounds,
            debug_one_epoch=args.debug_one_epoch,
        )
    elif _is_bounded_hier_shared_gaussian_vm_d(cfg):
        if va_transform is None or cal_a is None:
            raise ValueError("bounded_hier model requires va_transform and cal_a from load_all_u_data")
        train_meta = train_stage_u_bounded_hier_shared_gaussian_vonmises(
            cfg,
            train_a,
            cal_a,
            vocabs,
            device=device,
            out_dir=run_dir,
            va_transform=va_transform,
            debug_one_epoch=args.debug_one_epoch,
        )
    elif _is_shared_gaussian_vm_d(cfg):
        train_meta = train_stage_u_shared_gaussian_vonmises(
            cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir
        )
    elif _is_hybrid_va_gmm_d(cfg):
        train_meta = train_stage_u_hybrid_gmm_d(cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir)
    else:
        train_meta = train_stage_u(cfg, train_a, cal_a, vocabs, device=device, out_dir=run_dir)
    wall = time.time() - t0

    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    is_hybrid = ckpt.get("head_family") == "hybrid_va_gmm_d"
    is_shared_vm = ckpt.get("head_family") == "shared_gaussian_vm_d"
    is_frozen_bounded = ckpt.get("head_family") == FROZEN_BOUNDED_SUPPORT_HEAD
    is_bounded_hier = ckpt.get("head_family") == "bounded_hier_shared_gaussian_vm_d"
    is_bounded_player = (
        ckpt.get("head_family") == "bounded_player_support_hier_shared_gaussian_vm_d"
    )
    T_cal = None
    T_va = T_mix = None
    T_ang = None
    if is_bounded_hier or is_bounded_player or is_shared_vm or is_frozen_bounded:
        T_va = float(ckpt["temperature_T_va"])
        T_ang = float(ckpt["temperature_T_ang"])
    elif is_hybrid:
        T_va = float(ckpt["temperature_T_va"])
        T_mix = float(ckpt["temperature_T_mix"])
    else:
        T_cal = float(ckpt["temperature_T"])

    num_f = train_a.x_num.shape[1]
    model = _build_model(cfg, num_f, vocabs, device)
    model.load_state_dict(ckpt["model_state"])

    if (is_bounded_player or is_frozen_bounded) and "target_transform" in ckpt:
        va_transform = PlayerSpecificBoundedLogitZScoreTransform.from_state_dict(
            ckpt["target_transform"]
        )
    elif is_bounded_hier and "target_transform" in ckpt:
        va_transform = BoundedLogitZScoreTransform.from_state_dict(ckpt["target_transform"])
    elif train_a.va_transform is not None:
        va_transform = train_a.va_transform
    else:
        va_transform = None

    tm = torch.tensor(train_a.target_means, dtype=torch.float32, device=device)
    ts = torch.tensor(train_a.target_stds, dtype=torch.float32, device=device)
    evcfg = cfg.get("evaluation", {})
    n_mc_d = int(evcfg.get("n_mc_coverage_draws_d", 256))
    bounds_eval = None
    if isinstance(va_transform, BoundedLogitZScoreTransform):
        bounds_eval = {k: tuple(v) for k, v in va_transform.bounds.items()}

    eval_splits = (
        [("train", train_a), ("val_select", val_a), ("temp_cal", temp_cal_a), ("test", test_a)]
        if (is_bounded_player or is_frozen_bounded)
        else [("train", train_a), ("calibration", cal_a), ("test", test_a)]
    )

    results: dict[str, dict] = {}
    for name, arr in eval_splits:
        if is_bounded_hier or is_bounded_player:
            x, cat, y, yr, w, bid = _arrays_to_tensors_bounded(arr, device)
            for label, tva, tang in [
                (f"{name}_pre_temp", 1.0, 1.0),
                (f"{name}_post_temp", T_va, T_ang),
            ]:
                pack = evaluate_bounded_hier_shared_gaussian_vm_split_batched(
                    model,
                    x,
                    cat,
                    bid,
                    y,
                    yr,
                    w,
                    va_transform,
                    T_va=tva,
                    T_ang=tang,
                    batch_size=4096,
                    device=device,
                    n_mc_circle=n_mc_d,
                    kappa_max=float(ckpt.get("kappa_max", 120.0)),
                    report_player_diagnostics=bool(evcfg.get("report_player_diagnostics", True)),
                    report_mixture_diagnostics=bool(evcfg.get("report_mixture_diagnostics", True)),
                    report_covariance_diagnostics=bool(evcfg.get("report_covariance_diagnostics", True)),
                    bounds=bounds_eval,
                )
                save_metrics_json_csv(run_dir / "metrics", label, pack)
                results[label] = pack
            pit_pack = evaluate_bounded_hier_shared_gaussian_vm_split_batched(
                model,
                x,
                cat,
                bid,
                y,
                yr,
                w,
                va_transform,
                T_va=T_va,
                T_ang=T_ang,
                batch_size=4096,
                device=device,
                n_mc_circle=n_mc_d,
                kappa_max=float(ckpt.get("kappa_max", 120.0)),
                return_pit_array=True,
            )
            pit_arr = pit_pack.get("pit_values")
            if pit_arr is not None:
                plot_pit_hist(pit_arr, run_dir / "plots" / f"pit_{name}_post_temp.png")
            continue

        x, cat, y, yr, w = _arrays_to_tensors(arr, device)
        if is_shared_vm or is_frozen_bounded:
            for label, tva, tang in [
                (f"{name}_pre_temp", 1.0, 1.0),
                (f"{name}_post_temp", T_va, T_ang),
            ]:
                bid_eval = (
                    torch.from_numpy(arr.batter_id).long().to(device)
                    if is_frozen_bounded and arr.batter_id is not None
                    else None
                )
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
                    va_transform=va_transform if is_frozen_bounded else None,
                    player_ids=bid_eval,
                )
                save_metrics_json_csv(run_dir / "metrics", label, pack)
                results[label] = pack
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
                save_metrics_json_csv(run_dir / "metrics", label, pack)
                results[label] = pack
        else:
            for label, T_use in [(f"{name}_pre_temp", 1.0), (f"{name}_post_temp", T_cal)]:
                pack = evaluate_split_batched(
                    model, x, cat, y, yr, w, tm, ts, T=T_use, batch_size=8192, device=device
                )
                save_metrics_json_csv(run_dir / "metrics", label, pack)
                results[label] = pack

        with torch.no_grad():
            x, cat, y, yr, w = _arrays_to_tensors(arr, device)
            if is_shared_vm or is_frozen_bounded:
                bid_eval = (
                    torch.from_numpy(arr.batter_id).long().to(device)
                    if is_frozen_bounded and arr.batter_id is not None
                    else None
                )
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
                    va_transform=va_transform if is_frozen_bounded else None,
                    player_ids=bid_eval,
                )
                pit_arr = pit_pack.get("pit_values")
                if pit_arr is not None:
                    plot_pit_hist(pit_arr, run_dir / "plots" / f"pit_{name}_post_temp.png")
            else:
                pits = []
                bs = 8192
                for s in range(0, x.shape[0], bs):
                    e = min(s + bs, x.shape[0])
                    if is_hybrid:
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
                    else:
                        mu, L = model(x[s:e], {k: v[s:e] for k, v in cat.items()})
                        Ls = L * float(np.sqrt(T_cal))
                        var_m = torch.diagonal(
                            torch.bmm(Ls, Ls.transpose(-1, -2)), dim1=-2, dim2=-1
                        ).clamp_min(1e-12)
                        std_m = torch.sqrt(var_m)
                        z = (y[s:e] - mu) / std_m.clamp_min(1e-8)
                        pits.append(torch.special.ndtr(z).detach().cpu().numpy())
                pit_arr = np.concatenate(pits, axis=0)
                plot_pit_hist(pit_arr, run_dir / "plots" / f"pit_{name}_post_temp.png")

    run_manifest = {
        "timestamp_utc": stamp,
        "git_hash": _git_hash(),
        "config_path": str(args.config.resolve()),
        "feature_contract_id": frozen_contract.get("contract_id"),
        "feature_contract_path": str(contract_path),
        "feature_contract_copied_to_run": str(run_dir / "feature_contract_frozen.json"),
        "row_counts": {k: int(meta[k]) for k in meta if k.startswith("n_rows")},
        "event_counts": {k: int(meta[k]) for k in meta if k.startswith("n_events")},
        "draw_cap_per_event": cfg["training"]["max_draws_per_event"],
        "batch_events": cfg["training"]["batch_events"],
        "event_balanced_batching": True,
        "player_constant_stats_computed_train_only": meta.get("player_constant_stats_from_train_only", []),
        "training_wall_seconds": wall,
        "train_loop_seconds": train_meta.get("train_loop_seconds"),
        "train_total_seconds_including_calibration": train_meta.get(
            "train_total_seconds_including_calibration"
        ),
        "head_family": ckpt.get("head_family", "full_gaussian_cholesky"),
        "covariance_temperature_T": (T_cal if not is_hybrid and not is_shared_vm else None),
        "covariance_temperature_T_va": (T_va if (is_hybrid or is_shared_vm or is_frozen_bounded) else None),
        "covariance_temperature_T_mix": (T_mix if is_hybrid else None),
        "angular_temperature_T_ang": (
            T_ang if (is_shared_vm or is_bounded_hier or is_bounded_player or is_frozen_bounded) else None
        ),
        "target_transform_type": meta.get("target_transform_type"),
        "nll_val_select_weighted_pre_temp": results.get("val_select_pre_temp", {}).get("weighted_nll"),
        "nll_val_select_weighted_post_temp": results.get("val_select_post_temp", {}).get("weighted_nll"),
        "nll_temp_cal_weighted_pre_temp": results.get("temp_cal_pre_temp", {}).get("weighted_nll"),
        "nll_temp_cal_weighted_post_temp": results.get("temp_cal_post_temp", {}).get("weighted_nll"),
        "nll_calibration_weighted_pre_temp": results.get("calibration_pre_temp", {}).get("weighted_nll"),
        "nll_calibration_weighted_post_temp": results.get("calibration_post_temp", {}).get("weighted_nll"),
        "nll_test_weighted_post_temp": results.get("test_post_temp", {}).get("weighted_nll"),
    }
    if run_manifest.get("nll_temp_cal_weighted_post_temp") is None:
        run_manifest["nll_temp_cal_weighted_post_temp"] = results.get("temp_cal_post_temp", {}).get(
            "weighted_nll"
        )
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    if is_shared_vm or is_bounded_hier or is_bounded_player or is_frozen_bounded:
        temp_line = f"- Temperatures: **T_va={T_va}**, **T_ang={T_ang}** (Gaussian × von Mises)"
    elif is_hybrid:
        temp_line = f"- Temperatures: **T_va={T_va}**, **T_mix={T_mix}** (hybrid head)"
    else:
        temp_line = f"- Temperature T (covariance scaling): **{T_cal}**"

    def _fmt(x):
        return f"{x:.4f}" if x is not None else "n/a"

    report_lines = [
        "# Stage-u training report",
        "",
        f"- Run directory: `{run_dir}`",
        temp_line,
        f"- val_select NLL pre/post T: **{_fmt(run_manifest.get('nll_val_select_weighted_pre_temp'))}** / **{_fmt(run_manifest.get('nll_val_select_weighted_post_temp'))}**",
        f"- temp_cal NLL pre/post T: **{_fmt(run_manifest.get('nll_temp_cal_weighted_pre_temp'))}** / **{_fmt(run_manifest.get('nll_temp_cal_weighted_post_temp'))}**",
        f"- Test weighted NLL (post T): **{_fmt(run_manifest.get('nll_test_weighted_post_temp'))}**",
        "",
        "## Metrics JSON/CSV",
        "",
        "See `metrics/` subdirectory.",
        "",
        "## Feature manifest",
        "",
        "`feature_manifest.json`",
        "",
    ]
    (project_root / "reports" / "p_u_given_g_training_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print("Done. Outputs:", run_dir)


if __name__ == "__main__":
    main()
