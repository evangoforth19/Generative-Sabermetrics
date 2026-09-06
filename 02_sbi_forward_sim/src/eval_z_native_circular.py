"""Stage-z hybrid Gaussian + von Mises: raw-space metrics, wrapped angular errors, sample PIT/coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .calibration_z import _hybrid_scaled_L
from .eval_z import save_metrics_json_csv
from .models_z import mdn_hybrid_gauss_vm_log_prob, sample_mdn_hybrid_gauss_vm
from .target_transform_z import wrap_deg


def weighted_hybrid_nll_pack(
    logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    mu_theta: torch.Tensor,
    kappa_theta: torch.Tensor,
    y_g: torch.Tensor,
    psi: torch.Tensor,
    theta: torch.Tensor,
    w: torch.Tensor,
    *,
    T_gauss: float = 1.0,
    T_ang: float = 1.0,
    T_gauss_x: float | None = None,
    T_gauss_y: float | None = None,
) -> tuple[float, float]:
    if T_gauss_x is not None and T_gauss_y is not None:
        tx = L_g.new_tensor(float(T_gauss_x))
        ty = L_g.new_tensor(float(T_gauss_y))
        lcal = _hybrid_scaled_L(L_g, tx, ty)
    else:
        scale = float(np.sqrt(T_gauss))
        lcal = L_g * scale
    kpsc = kappa_psi / float(T_ang)
    kthc = kappa_theta / float(T_ang)
    lp = mdn_hybrid_gauss_vm_log_prob(y_g, psi, theta, logits, mu_g, lcal, mu_psi, kpsc, mu_theta, kthc)
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    return float((nll * w).sum().item() / wsum), float(nll.mean().item())


def _pit_linear(samples: np.ndarray, true_v: float) -> float:
    s = samples.astype(np.float64)
    u = (np.sum(s < true_v) + 0.5 * np.sum(s == true_v)) / max(len(s), 1)
    return float(np.clip(u, 1e-6, 1 - 1e-6))


def _pit_angle(samples_deg: np.ndarray, true_deg: float) -> float:
    d = wrap_deg(samples_deg.astype(np.float64) - true_deg)
    u = float(np.mean(d < 0) + 0.5 * np.mean(d == 0))
    return float(np.clip(u, 1e-6, 1 - 1e-6))


def _in_linear_interval(samples: np.ndarray, true_v: float, lo: float, hi: float) -> bool:
    a, b = np.percentile(samples, [lo, hi])
    return bool(a <= true_v <= b)


def _in_angle_interval(samples_deg: np.ndarray, true_deg: float, lo: float, hi: float) -> bool:
    from scipy.stats import circmean

    rad_s = np.radians(samples_deg)
    mu = circmean(rad_s, high=np.pi, low=-np.pi)
    rel_s = np.angle(np.exp(1j * (rad_s - mu)))
    rel_t = float(np.angle(np.exp(1j * (np.radians(true_deg) - mu))))
    lo_r, hi_r = np.percentile(rel_s, [lo, hi])
    return bool(lo_r <= rel_t <= hi_r)


def _mixture_mean_gauss(pi: torch.Tensor, mu_g: torch.Tensor) -> torch.Tensor:
    return (pi.unsqueeze(-1) * mu_g).sum(dim=1)


def _mixture_mean_angle(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    s = (pi * torch.sin(mu)).sum(dim=-1)
    c = (pi * torch.cos(mu)).sum(dim=-1)
    return torch.atan2(s, c)


@torch.no_grad()
def evaluate_z_native_circular_split_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_gauss: torch.Tensor,
    psi_rad: torch.Tensor,
    theta_rad: torch.Tensor,
    y_raw_four: torch.Tensor,
    w: torch.Tensor,
    gauss_means: torch.Tensor,
    gauss_stds: torch.Tensor,
    *,
    T_gauss: float = 1.0,
    T_ang: float = 1.0,
    T_gauss_x: float | None = None,
    T_gauss_y: float | None = None,
    n_samples: int = 256,
    batch_size: int = 512,
    device: torch.device,
    seed: int = 0,
) -> dict[str, Any]:
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    n = x_num.shape[0]
    gm = gauss_means.view(1, 2).to(device=device, dtype=torch.float32)
    gs = gauss_stds.view(1, 2).to(device=device, dtype=torch.float32)

    sum_w_nll = 0.0
    sum_u_nll = 0.0
    sum_w = 0.0

    sum_abs_x = 0.0
    sum_abs_ey = 0.0
    sum_sq_x = 0.0
    sum_sq_ey = 0.0

    pits_x: list[float] = []
    pits_ey: list[float] = []
    pits_psi: list[float] = []
    pits_theta: list[float] = []

    cnt_c50 = {"x": 0, "e_y_star": 0, "psi_deg": 0, "theta_deg": 0}
    cnt_c80 = {"x": 0, "e_y_star": 0, "psi_deg": 0, "theta_deg": 0}
    cnt_c90 = {"x": 0, "e_y_star": 0, "psi_deg": 0, "theta_deg": 0}

    w_abs_psi = 0.0
    w_abs_th = 0.0
    w_sq_psi = 0.0
    w_sq_th = 0.0

    y_rf = y_raw_four.cpu().numpy()
    w_np = w.cpu().numpy()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        xb = x_num[sl]
        cb = {k: v[sl] for k, v in cat.items()}
        ygb = y_gauss[sl]
        pb = psi_rad[sl]
        tb = theta_rad[sl]
        wb = w[sl]
        b = end - start

        logits, mu_g, L_g, mu_psi, kappa_psi, mu_theta, kappa_theta = model(xb, cb)
        if T_gauss_x is not None and T_gauss_y is not None:
            tx = L_g.new_tensor(float(T_gauss_x))
            ty = L_g.new_tensor(float(T_gauss_y))
            lcal = _hybrid_scaled_L(L_g, tx, ty)
        else:
            scale = float(np.sqrt(T_gauss))
            lcal = L_g * scale
        kpsc = kappa_psi / float(T_ang)
        kthc = kappa_theta / float(T_ang)
        lp = mdn_hybrid_gauss_vm_log_prob(ygb, pb, tb, logits, mu_g, lcal, mu_psi, kpsc, mu_theta, kthc)
        nll = -lp
        sum_w_nll += float((nll * wb).sum().item())
        sum_u_nll += float(nll.sum().item())
        sum_w += float(wb.sum().item())

        pi = F.softmax(logits, dim=-1)
        mu_g_pred = _mixture_mean_gauss(pi, mu_g)
        gmx, gmy = gm[0, 0].item(), gm[0, 1].item()
        gsx, gsy = gs[0, 0].item(), gs[0, 1].item()
        x_m = (mu_g_pred[:, 0] * gsx + gmx).cpu().numpy()
        ey_m = (mu_g_pred[:, 1] * gsy + gmy).cpu().numpy()
        psi_m_deg = np.degrees(_mixture_mean_angle(pi, mu_psi).cpu().numpy())
        th_m_deg = np.degrees(_mixture_mean_angle(pi, mu_theta).cpu().numpy())

        x_t = y_rf[sl, 0]
        psi_t = y_rf[sl, 1]
        ey_t = y_rf[sl, 2]
        th_t = y_rf[sl, 3]
        w_b = w_np[sl]

        sum_abs_x += float(np.sum(np.abs(x_m - x_t) * w_b))
        sum_abs_ey += float(np.sum(np.abs(ey_m - ey_t) * w_b))
        sum_sq_x += float(np.sum((x_m - x_t) ** 2 * w_b))
        sum_sq_ey += float(np.sum((ey_m - ey_t) ** 2 * w_b))

        dw = wrap_deg(psi_m_deg - psi_t)
        dth = wrap_deg(th_m_deg - th_t)
        w_abs_psi += float(np.sum(np.abs(dw) * w_b))
        w_abs_th += float(np.sum(np.abs(dth) * w_b))
        w_sq_psi += float(np.sum((dw**2) * w_b))
        w_sq_th += float(np.sum((dth**2) * w_b))

        samp_g, samp_psi, samp_th = sample_mdn_hybrid_gauss_vm(
            logits, mu_g, lcal, mu_psi, kpsc, mu_theta, kthc, n_samples, generator=g
        )
        sx = (samp_g[..., 0] * gsx + gmx).cpu().numpy()
        sey = (samp_g[..., 1] * gsy + gmy).cpu().numpy()
        psi_s = np.degrees(samp_psi.cpu().numpy())
        th_s = np.degrees(samp_th.cpu().numpy())

        for i in range(b):
            pits_x.append(_pit_linear(sx[i], float(x_t[i])))
            pits_ey.append(_pit_linear(sey[i], float(ey_t[i])))
            pits_psi.append(_pit_angle(psi_s[i], float(psi_t[i])))
            pits_theta.append(_pit_angle(th_s[i], float(th_t[i])))

            if _in_linear_interval(sx[i], float(x_t[i]), 25, 75):
                cnt_c50["x"] += 1
            if _in_linear_interval(sx[i], float(x_t[i]), 10, 90):
                cnt_c80["x"] += 1
            if _in_linear_interval(sx[i], float(x_t[i]), 5, 95):
                cnt_c90["x"] += 1

            if _in_linear_interval(sey[i], float(ey_t[i]), 25, 75):
                cnt_c50["e_y_star"] += 1
            if _in_linear_interval(sey[i], float(ey_t[i]), 10, 90):
                cnt_c80["e_y_star"] += 1
            if _in_linear_interval(sey[i], float(ey_t[i]), 5, 95):
                cnt_c90["e_y_star"] += 1

            if _in_angle_interval(psi_s[i], float(psi_t[i]), 25, 75):
                cnt_c50["psi_deg"] += 1
            if _in_angle_interval(psi_s[i], float(psi_t[i]), 10, 90):
                cnt_c80["psi_deg"] += 1
            if _in_angle_interval(psi_s[i], float(psi_t[i]), 5, 95):
                cnt_c90["psi_deg"] += 1

            if _in_angle_interval(th_s[i], float(th_t[i]), 25, 75):
                cnt_c50["theta_deg"] += 1
            if _in_angle_interval(th_s[i], float(th_t[i]), 10, 90):
                cnt_c80["theta_deg"] += 1
            if _in_angle_interval(th_s[i], float(th_t[i]), 5, 95):
                cnt_c90["theta_deg"] += 1

    inv = 1.0 / max(n, 1)
    inv_w = 1.0 / max(sum_w, 1e-12)
    pit_x = np.array(pits_x)
    pit_psi = np.array(pits_psi)
    pit_th = np.array(pits_theta)
    pit_ey = np.array(pits_ey)

    return {
        "n_rows": int(n),
        "weighted_nll_hybrid_natural_space": sum_w_nll / max(sum_w, 1e-12),
        "unweighted_nll_hybrid_natural_space": sum_u_nll / max(n, 1),
        "n_predictive_samples_per_row": int(n_samples),
        "note_nll": "Joint NLL: log sum_k pi_k N(y_g) VM(psi) VM(theta) in (std Gaussian block, rad angles); "
        "not comparable to 6D circular-GMM or 4D linear NLL scalars.",
        "mae_raw_x": sum_abs_x * inv_w,
        "rmse_raw_x": float(np.sqrt(sum_sq_x * inv_w)),
        "mae_raw_e_y_star": sum_abs_ey * inv_w,
        "rmse_raw_e_y_star": float(np.sqrt(sum_sq_ey * inv_w)),
        "wrapped_mae_deg_psi": w_abs_psi * inv_w,
        "wrapped_rmse_deg_psi": float(np.sqrt(w_sq_psi * inv_w)),
        "wrapped_mae_deg_theta": w_abs_th * inv_w,
        "wrapped_rmse_deg_theta": float(np.sqrt(w_sq_th * inv_w)),
        "point_estimate_note": "x/e_y*: mixture mean in standardized (x,e_y*) then destandardized; "
        "angles: circular mixture mean atan2(sum pi sin mu, sum pi cos mu) per row.",
        "coverage_50_marginal_sample": {k: cnt_c50[k] * inv for k in cnt_c50},
        "coverage_80_marginal_sample": {k: cnt_c80[k] * inv for k in cnt_c80},
        "coverage_90_marginal_sample": {k: cnt_c90[k] * inv for k in cnt_c90},
        "pit_hist_counts_x": np.histogram(pit_x, bins=20, range=(0, 1))[0].tolist(),
        "pit_hist_counts_e_y_star": np.histogram(pit_ey, bins=20, range=(0, 1))[0].tolist(),
        "pit_hist_counts_psi_deg": np.histogram(pit_psi, bins=20, range=(0, 1))[0].tolist(),
        "pit_hist_counts_theta_deg": np.histogram(pit_th, bins=20, range=(0, 1))[0].tolist(),
        "eval_note": "Angular PIT/coverage from native von Mises predictive samples (radians→deg); "
        "wrapped errors in degrees.",
        "_pit_arrays": {
            "x": pit_x,
            "e_y_star": pit_ey,
            "psi_deg": pit_psi,
            "theta_deg": pit_th,
        },
    }


def plot_pit_hist_native(pit_pack: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["x", "e_y_star", "psi_deg", "theta_deg"]
    keys = ["x", "ey", "psi", "theta"]
    arrs = [pit_pack[k] for k in keys]
    for ax, t, arr in zip(axes, titles, arrs, strict=True):
        ax.hist(arr, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax.set_title(t)
        ax.set_xlabel("PIT")
    fig.suptitle("Stage-z native circular (Gauss + von Mises): marginal PIT (sample-based, raw space)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
