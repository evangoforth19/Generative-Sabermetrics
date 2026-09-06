#!/usr/bin/env python3
"""Refit post-hoc temperatures only from an existing native-circular stage-z checkpoint (same weights)."""

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

from sbi_forward_sim.src.calibration_z import fit_hybrid_native_temperature  # noqa: E402
from sbi_forward_sim.src.data_z import load_all_z_data_native_circular  # noqa: E402
from sbi_forward_sim.src.eval_z_native_circular import (  # noqa: E402
    evaluate_z_native_circular_split_batched,
    plot_pit_hist_native,
    save_metrics_json_csv,
)
from sbi_forward_sim.src.feature_contract_z import load_feature_contract_z  # noqa: E402
from sbi_forward_sim.src.models_z import ConditionalHybridGaussVonMisesMixtureZ  # noqa: E402
from sbi_forward_sim.src.schema import Z_TARGET_COLUMNS  # noqa: E402


def _save_metrics_strip(pack: dict, out_dir: Path, name: str) -> None:
    p = dict(pack)
    p.pop("_pit_arrays", None)
    save_metrics_json_csv(out_dir / "metrics", name, p)


def _build_hybrid(cfg: dict, num_f: int, vocabs: dict[str, dict[str, int]], device: torch.device):
    mcfg = cfg["model"]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesMixtureZ(
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
    ).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        help="YAML with paths, calibration.anisotropic_gaussian, eval; defaults to source config_resolved",
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

    train_a, cal_a, test_a, bundle = load_all_z_data_native_circular(cfg, project_root)
    vocabs = bundle["vocabs"]
    device = torch.device("cpu")

    num_f = train_a.x_num.shape[1]
    model = _build_hybrid(cfg, num_f, vocabs, device)
    model.load_state_dict(old["model_state"])
    model.eval()

    x_cal = torch.from_numpy(cal_a.x_num).to(device)
    cat_cal = {k: torch.from_numpy(v).long().to(device) for k, v in cal_a.cat.items()}
    yg_cal = torch.from_numpy(cal_a.y_gauss).to(device)
    psi_cal = torch.from_numpy(cal_a.psi_rad).to(device)
    th_cal = torch.from_numpy(cal_a.theta_rad).to(device)
    w_cal = torch.from_numpy(cal_a.w).to(device)

    with torch.no_grad():
        lg, mg, Lg, mp, kp, mt, kt = model(x_cal, cat_cal)

    ccal = cfg["calibration"]
    htemp = fit_hybrid_native_temperature(
        lg.detach(),
        mg.detach(),
        Lg.detach(),
        mp.detach(),
        kp.detach(),
        mt.detach(),
        kt.detach(),
        yg_cal,
        psi_cal,
        th_cal,
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
        "Gaussian: row i of L scaled by sqrt(T_gauss_i) for x and e_y* "
        "(Σ_cal = D Σ D); T_gauss is geometric mean of T_x,T_y. "
        "Angular: kappa_cal = kappa / T_ang for ψ and θ."
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
    T_gauss = float(htemp.T_gauss)

    gm = torch.tensor(train_a.gauss_means, dtype=torch.float32, device=device)
    gs = torch.tensor(train_a.gauss_stds, dtype=torch.float32, device=device)
    n_samp = int(cfg.get("eval", {}).get("n_predictive_samples", 256))
    bs = int(cfg.get("eval", {}).get("circular_eval_batch_size", 512))
    eval_seed = int(cfg["training"].get("seed", 42))

    results: dict[str, dict] = {}
    for split_name, arr in [("train", train_a), ("calibration", cal_a), ("test", test_a)]:
        x = torch.from_numpy(arr.x_num).to(device)
        cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
        yg = torch.from_numpy(arr.y_gauss).to(device)
        psi = torch.from_numpy(arr.psi_rad).to(device)
        th = torch.from_numpy(arr.theta_rad).to(device)
        yr = torch.from_numpy(arr.y_raw_four).to(device)
        w = torch.from_numpy(arr.w).to(device)
        for label, tg, ta, tgx, tgy in [
            (f"{split_name}_pre_temp", 1.0, 1.0, None, None),
            (f"{split_name}_post_temp", T_gauss, T_ang, T_gx_f, T_gy_f),
        ]:
            pack = evaluate_z_native_circular_split_batched(
                model,
                x,
                cat,
                yg,
                psi,
                th,
                yr,
                w,
                gm,
                gs,
                T_gauss=tg,
                T_ang=ta,
                T_gauss_x=tgx,
                T_gauss_y=tgy,
                n_samples=n_samp,
                batch_size=bs,
                device=device,
                seed=eval_seed + {"train": 0, "calibration": 1, "test": 2}[split_name],
            )
            pits = pack.pop("_pit_arrays", None)
            _save_metrics_strip(pack, run_dir, label)
            results[label] = pack
            if pits is not None and label.endswith("_post_temp"):
                plot_pit_hist_native(
                    {
                        "x": pits["x"],
                        "ey": pits["e_y_star"],
                        "psi": pits["psi_deg"],
                        "theta": pits["theta_deg"],
                    },
                    run_dir / "plots" / f"pit_{split_name}_post_temp.png",
                )
                import numpy as np

                Path(run_dir).mkdir(exist_ok=True)
                np.savez_compressed(
                    run_dir / f"pits_{split_name}_post_temp.npz",
                    x=pits["x"],
                    e_y_star=pits["e_y_star"],
                    psi_deg=pits["psi_deg"],
                    theta_deg=pits["theta_deg"],
                )

    fc = load_feature_contract_z(cfg, project_root)
    meta = bundle["meta"]
    run_manifest = {
        "timestamp_utc": stamp,
        "config_path": str(cfg_path.resolve()),
        "feature_contract_id": fc.get("contract_id"),
        "target_parameterization": "native_circular_vm",
        "model_family": "hybrid_gauss_vonmises_shared",
        "row_counts": {k: int(meta[k]) for k in meta if k.startswith("n_rows")},
        "event_counts": {k: int(meta[k]) for k in meta if k.startswith("n_events")},
        "temperature_T_gauss": T_gauss,
        "temperature_T_gauss_x": T_gx_f,
        "temperature_T_gauss_y": T_gy_f,
        "temperature_T_ang": T_ang,
        "recalibrated_from_checkpoint": str(src / "checkpoint.pt"),
        "nll_test_hybrid_weighted_post_temp": results.get("test_post_temp", {}).get(
            "weighted_nll_hybrid_natural_space"
        ),
        "eval_predictive_samples": n_samp,
        "raw_target_order": list(Z_TARGET_COLUMNS),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("Done. Outputs:", run_dir)


if __name__ == "__main__":
    main()
