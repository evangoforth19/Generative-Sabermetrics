"""Rich evaluation for branch_autoreg_bounded_hybrid_mdn_z."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .calibration_z import BranchAutoregTemperatureScale
from .data_z_branch_autoreg import ZArraysBranchAutoreg
from .eval_z import save_metrics_json_csv
from .eval_z_native_circular import _in_angle_interval, _in_linear_interval, _pit_angle, _pit_linear
from .models_z import BranchAutoregBoundedHybridMDNZ, sample_branch_autoreg_bounded_mdn
from .target_transforms_z import model_to_raw_coords
from .train_z_branch_autoreg import weighted_eval_nll


def _temp_from_dict(ckpt_or: dict[str, Any] | BranchAutoregTemperatureScale | None) -> BranchAutoregTemperatureScale:
    if ckpt_or is None:
        return BranchAutoregTemperatureScale(1.0, 1.0, 1.0, 1.0)
    if isinstance(ckpt_or, BranchAutoregTemperatureScale):
        return ckpt_or
    return BranchAutoregTemperatureScale(
        float(ckpt_or.get("T_x", ckpt_or.get("temperature_T_x", 1.0))),
        float(ckpt_or.get("T_y", ckpt_or.get("temperature_T_y", 1.0))),
        float(ckpt_or.get("T_psi", ckpt_or.get("temperature_T_psi", 1.0))),
        float(ckpt_or.get("T_ex", ckpt_or.get("temperature_T_ex", 1.0))),
    )


@torch.no_grad()
def evaluate_branch_autoreg_split(
    model: BranchAutoregBoundedHybridMDNZ,
    arr: ZArraysBranchAutoreg,
    device: torch.device,
    *,
    temp: BranchAutoregTemperatureScale | None = None,
    n_samples: int = 256,
    batch_size: int = 512,
    seed: int = 0,
) -> dict[str, Any]:
    model.eval()
    temp = _temp_from_dict(temp)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    st = arr.transform_state

    n = arr.x_num.shape[0]
    x_all = torch.from_numpy(arr.x_num).to(device)
    cat_all = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    rxy_all = torch.from_numpy(arr.r_xy_std).to(device)
    psi_all = torch.from_numpy(arr.psi_rad).to(device)
    rex_all = torch.from_numpy(arr.r_ex_std).to(device)
    j_all = torch.from_numpy(arr.j_log_raw).to(device)
    w_all = torch.from_numpy(arr.w).to(device)
    y_raw = arr.y_raw

    w_model = weighted_eval_nll(model, arr, device, batch_size=batch_size, temp=temp)
    w_raw = 0.0
    wsum = 0.0
    pits = {k: [] for k in ("x", "e_y_star", "psi_deg", "e_x")}
    cov50 = {k: 0 for k in pits}
    cov80 = {k: 0 for k in pits}
    cov90 = {k: 0 for k in pits}
    k_eff_list: list[float] = []
    branch_usage = np.zeros(model.n_components, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        bx = x_all[sl]
        bc = {k: v[sl] for k, v in cat_all.items()}
        h = model.trunk(bx, bc)
        logits, mu_xy, L_xy = model.mixture_params(h)
        lp_m, _ = model.log_prob_components(
            rxy_all[sl],
            psi_all[sl],
            rex_all[sl],
            logits,
            mu_xy,
            L_xy,
            h,
            T_x=temp.T_x,
            T_y=temp.T_y,
            T_psi=temp.T_psi,
            T_ex=temp.T_ex,
        )
        lp_r = lp_m + j_all[sl]
        wx = w_all[sl]
        wsum += float(wx.sum().item())
        w_raw += float(((-lp_r) * wx).sum().item())

        pi = F.softmax(logits, dim=-1)
        k_eff = 1.0 / (pi * pi).sum(dim=-1).clamp_min(1e-8)
        k_eff_list.extend(k_eff.cpu().numpy().tolist())
        branch_usage += (pi.sum(dim=0).cpu().numpy() * wx.sum().item())

        r_s, p_s, e_s = sample_branch_autoreg_bounded_mdn(
            model,
            bx,
            bc,
            n_samples=n_samples,
            T_x=temp.T_x,
            T_y=temp.T_y,
            T_psi=temp.T_psi,
            T_ex=temp.T_ex,
            generator=g,
        )
        x_lo = arr.x_lower[sl]
        x_hi = arr.x_upper[sl]
        for i in range(end - start):
            xi = int(start + i)
            x_t, ey_t, ex_t = float(y_raw[xi, 0]), float(y_raw[xi, 2]), float(y_raw[xi, 3])
            psi_t = float(y_raw[xi, 1])
            x_s = []
            ey_s = []
            ex_s = []
            for si in range(n_samples):
                xr, eyr, exr = model_to_raw_coords(
                    float(r_s[i, si, 0].cpu()),
                    float(r_s[i, si, 1].cpu()),
                    float(e_s[i, si].cpu()),
                    x_lower=np.array([x_lo[i]]),
                    x_upper=np.array([x_hi[i]]),
                    state=st,
                )
                x_s.append(float(np.asarray(xr).reshape(-1)[0]))
                ey_s.append(float(np.asarray(eyr).reshape(-1)[0]))
                ex_s.append(float(np.asarray(exr).reshape(-1)[0]))
            x_s = np.array(x_s)
            ey_s = np.array(ey_s)
            ex_s = np.array(ex_s)
            psi_s = np.degrees(p_s[i].cpu().numpy())
            pits["x"].append(_pit_linear(x_s, x_t))
            pits["e_y_star"].append(_pit_linear(ey_s, ey_t))
            pits["e_x"].append(_pit_linear(ex_s, ex_t))
            pits["psi_deg"].append(_pit_angle(psi_s, psi_t))
            for lvl, cnt in ((50, cov50), (80, cov80), (90, cov90)):
                if _in_linear_interval(x_s, x_t, 100 - lvl, lvl):
                    cnt["x"] += 1
                if _in_linear_interval(ey_s, ey_t, 100 - lvl, lvl):
                    cnt["e_y_star"] += 1
                if _in_linear_interval(ex_s, ex_t, 100 - lvl, lvl):
                    cnt["e_x"] += 1
                if _in_angle_interval(psi_s, psi_t, 100 - lvl, lvl):
                    cnt["psi_deg"] += 1

    inv = 1.0 / max(n, 1)
    ke = np.array(k_eff_list)
    return {
        "n_rows": int(n),
        "weighted_joint_nll_model_space": w_model,
        "weighted_joint_nll_raw_space_with_jacobian": w_raw / max(wsum, 1e-12),
        "n_predictive_samples_per_row": int(n_samples),
        "coverage_50_marginal_sample": {k: cov50[k] * inv for k in cov50},
        "coverage_80_marginal_sample": {k: cov80[k] * inv for k in cov80},
        "coverage_90_marginal_sample": {k: cov90[k] * inv for k in cov90},
        "K_eff_mean": float(ke.mean()),
        "K_eff_median": float(np.median(ke)),
        "K_eff_p5": float(np.percentile(ke, 5)),
        "K_eff_p95": float(np.percentile(ke, 95)),
        "branch_usage_mean": (branch_usage / max(wsum, 1e-12)).tolist(),
        "_pit_arrays": {k: np.array(v) for k, v in pits.items()},
    }


def plot_pit_hist_branch(pit_pack: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, key in zip(axes, ["x", "e_y_star", "psi_deg", "e_x"], strict=True):
        arr = pit_pack.get(key, np.array([]))
        ax.hist(arr, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax.set_title(key)
    fig.suptitle("Stage-z branch autoreg bounded MDN: marginal PIT")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_rich_eval_all_splits(
    model: BranchAutoregBoundedHybridMDNZ,
    splits: dict[str, ZArraysBranchAutoreg],
    device: torch.device,
    out_dir: Path,
    *,
    temp: BranchAutoregTemperatureScale,
    cfg: dict[str, Any],
) -> dict[str, dict]:
    n_samp = int(cfg.get("evaluation", {}).get("n_predictive_samples", 256))
    bs = int(cfg.get("evaluation", {}).get("eval_batch_size", 512))
    results: dict[str, dict] = {}
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    for name, arr in splits.items():
        pack = evaluate_branch_autoreg_split(
            model, arr, device, temp=temp, n_samples=n_samp, batch_size=bs, seed=42 + hash(name) % 1000
        )
        pits = pack.pop("_pit_arrays", {})
        save_metrics_json_csv(metrics_dir, f"{name}_post_temp", pack)
        if pits:
            plot_pit_hist_branch(pits, plots_dir / f"pit_{name}_post_temp.png")
        results[name] = pack
    return results
