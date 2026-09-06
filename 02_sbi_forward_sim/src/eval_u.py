"""Metrics and diagnostics for stage-u Gaussian model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

U_NAMES = ("v_ss_tilde", "a_tilde", "d_tilde")


def weighted_shared_gaussian_vm_nll_torch(
    mu_va: torch.Tensor,
    scale_tril_va: torch.Tensor,
    mix_logits: torch.Tensor,
    loc_d: torch.Tensor,
    kappa_d: torch.Tensor,
    y_std: torch.Tensor,
    y_raw_d: torch.Tensor,
    w: torch.Tensor,
    *,
    T_va: float = 1.0,
    T_ang: float = 1.0,
    kappa_max: float = 120.0,
) -> tuple[float, float]:
    """Joint NLL for shared Gaussian–von Mises mixture (standardized va, raw-degree d)."""
    from .models_u import shared_mixture_bivariate_gaussian_vonmises_log_prob

    L = scale_tril_va * float(np.sqrt(T_va))
    kc = (kappa_d / float(T_ang)).clamp(min=1e-4, max=float(kappa_max))
    d_rad = torch.deg2rad(y_raw_d)
    lp = shared_mixture_bivariate_gaussian_vonmises_log_prob(
        y_std[:, :2],
        d_rad,
        mix_logits,
        mu_va,
        L,
        loc_d,
        kc,
        kappa_max=kappa_max,
    )
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    wmean = float((nll * w).sum().item() / wsum)
    umean = float(nll.mean().item())
    return wmean, umean


def weighted_hybrid_joint_nll_torch(
    mu_va: torch.Tensor,
    scale_tril_va: torch.Tensor,
    mix_logits: torch.Tensor,
    mix_mu: torch.Tensor,
    mix_scale: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    T_va: float = 1.0,
    T_mix: float = 1.0,
) -> tuple[float, float]:
    """Joint NLL for hybrid head (standardized y)."""
    from .models_u import mixture_log_prob_1d

    L = scale_tril_va * float(np.sqrt(T_va))
    dist = torch.distributions.MultivariateNormal(mu_va, scale_tril=L, validate_args=False)
    lp2 = dist.log_prob(y[:, :2])
    sig = mix_scale * float(np.sqrt(T_mix))
    lpd = mixture_log_prob_1d(y[:, 2], mix_logits, mix_mu, sig)
    lp = lp2 + lpd
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    wmean = float((nll * w).sum().item() / wsum)
    umean = float(nll.mean().item())
    return wmean, umean


def weighted_gaussian_nll_torch(
    mu: torch.Tensor,
    scale_tril: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    T: float = 1.0,
) -> tuple[float, float]:
    L = scale_tril * float(np.sqrt(T))
    dist = torch.distributions.MultivariateNormal(mu, scale_tril=L, validate_args=False)
    lp = dist.log_prob(y)
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    wmean = float((nll * w).sum().item() / wsum)
    umean = float(nll.mean().item())
    return wmean, umean


@torch.no_grad()
def evaluate_split_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_std: torch.Tensor,
    y_raw: torch.Tensor,
    w: torch.Tensor,
    target_means: torch.Tensor,
    target_stds: torch.Tensor,
    *,
    T: float = 1.0,
    batch_size: int = 8192,
    device: torch.device,
) -> dict[str, Any]:
    """Aggregate metrics over a split; tensors already on ``device``."""
    model.eval()
    n = x_num.shape[0]
    tm = target_means.view(1, 3)
    ts = target_stds.view(1, 3)

    sum_w_nll = 0.0
    sum_u_nll = 0.0
    sum_w = 0.0
    sum_abs_err = torch.zeros(3, device=device)
    sum_sq_err = torch.zeros(3, device=device)
    cnt_cov50 = torch.zeros(3, device=device)
    cnt_cov80 = torch.zeros(3, device=device)
    cnt_cov90 = torch.zeros(3, device=device)
    sum_width50 = torch.zeros(3, device=device)
    sum_width80 = torch.zeros(3, device=device)
    sum_width90 = torch.zeros(3, device=device)
    sum_pred_var_raw = torch.zeros(3, device=device)
    sum_corr_mat = torch.zeros(3, 3, device=device)
    pits_store: list[np.ndarray] = []

    z50 = float(stats.norm.ppf(0.75))
    z80 = float(stats.norm.ppf(0.9))
    z90 = float(stats.norm.ppf(0.95))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        xb = x_num[sl]
        cb = {k: v[sl] for k, v in cat.items()}
        mu, L = model(xb, cb)
        Ls = L * float(np.sqrt(T))
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=Ls, validate_args=False)
        lp = dist.log_prob(y_std[sl])
        nll = -lp
        wb = w[sl]
        sum_w_nll += float((nll * wb).sum().item())
        sum_u_nll += float(nll.sum().item())
        sum_w += float(wb.sum().item())

        Sig = torch.bmm(Ls, Ls.transpose(-1, -2))
        var_marg = torch.diagonal(Sig, dim1=-2, dim2=-1).clamp_min(1e-12)
        std_marg = torch.sqrt(var_marg)

        mu_raw = mu * ts + tm
        yb = y_raw[sl]
        err = yb - mu_raw
        sum_abs_err += torch.sum(torch.abs(err), dim=0)
        sum_sq_err += torch.sum(err**2, dim=0)

        for k in range(3):
            low50 = (mu[:, k] - z50 * std_marg[:, k]) * ts[0, k] + tm[0, k]
            high50 = (mu[:, k] + z50 * std_marg[:, k]) * ts[0, k] + tm[0, k]
            low80 = (mu[:, k] - z80 * std_marg[:, k]) * ts[0, k] + tm[0, k]
            high80 = (mu[:, k] + z80 * std_marg[:, k]) * ts[0, k] + tm[0, k]
            low90 = (mu[:, k] - z90 * std_marg[:, k]) * ts[0, k] + tm[0, k]
            high90 = (mu[:, k] + z90 * std_marg[:, k]) * ts[0, k] + tm[0, k]
            yk = yb[:, k]
            cnt_cov50[k] += float(((yk >= low50) & (yk <= high50)).sum().item())
            cnt_cov80[k] += float(((yk >= low80) & (yk <= high80)).sum().item())
            cnt_cov90[k] += float(((yk >= low90) & (yk <= high90)).sum().item())
            sum_width50[k] += float((high50 - low50).sum().item())
            sum_width80[k] += float((high80 - low80).sum().item())
            sum_width90[k] += float((high90 - low90).sum().item())

        sum_pred_var_raw += torch.sum(var_marg * (ts**2), dim=0)

        inv_std = 1.0 / std_marg.clamp_min(1e-8)
        z = (y_std[sl] - mu) * inv_std
        pit = torch.special.ndtr(z).cpu().numpy()
        pits_store.append(pit)

        std_rows = torch.sqrt(torch.diagonal(Sig, dim1=-2, dim2=-1).clamp_min(1e-12))
        corr_b = Sig / (std_rows[:, :, None] * std_rows[:, None, :] + 1e-12)
        sum_corr_mat += torch.sum(corr_b, dim=0)

    inv_total = 1.0 / float(max(n, 1))
    wmean_nll = sum_w_nll / max(sum_w, 1e-12)
    umean_nll = sum_u_nll / max(n, 1)
    mae = (sum_abs_err * inv_total).cpu().numpy().tolist()
    rmse = torch.sqrt(sum_sq_err * inv_total).cpu().numpy().tolist()
    cov50 = (cnt_cov50 * inv_total).cpu().numpy().tolist()
    cov80 = (cnt_cov80 * inv_total).cpu().numpy().tolist()
    cov90 = (cnt_cov90 * inv_total).cpu().numpy().tolist()
    width50 = (sum_width50 * inv_total).cpu().numpy().tolist()
    width80 = (sum_width80 * inv_total).cpu().numpy().tolist()
    width90 = (sum_width90 * inv_total).cpu().numpy().tolist()
    avg_pred_std_raw = torch.sqrt(sum_pred_var_raw * inv_total).cpu().numpy().tolist()
    mean_corr = (sum_corr_mat * inv_total).cpu().numpy()
    pits = np.concatenate(pits_store, axis=0)
    empirical_std_raw = np.std(y_raw.cpu().numpy(), axis=0).tolist()
    hist_counts = [np.histogram(pits[:, k], bins=20, range=(0, 1))[0].tolist() for k in range(3)]

    return {
        "n_rows": int(n),
        "weighted_nll": wmean_nll,
        "unweighted_nll": umean_nll,
        "mae_raw": mae,
        "rmse_raw": rmse,
        "target_names": list(U_NAMES),
        "coverage_50_marginal": cov50,
        "coverage_80_marginal": cov80,
        "coverage_90_marginal": cov90,
        "interval_width_50_raw_mean": width50,
        "interval_width_80_raw_mean": width80,
        "interval_width_90_raw_mean": width90,
        "avg_pred_std_marginal_raw": avg_pred_std_raw,
        "empirical_std_raw": empirical_std_raw,
        "mean_predicted_corr": mean_corr.tolist(),
        "pit_hist_counts": hist_counts,
    }


@torch.no_grad()
def evaluate_hybrid_split_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_std: torch.Tensor,
    y_raw: torch.Tensor,
    w: torch.Tensor,
    target_means: torch.Tensor,
    target_stds: torch.Tensor,
    *,
    T_va: float = 1.0,
    T_mix: float = 1.0,
    batch_size: int = 8192,
    device: torch.device,
) -> dict[str, Any]:
    """
    Metrics for HybridBivariateGaussianMixtureDNet.

    Marginals (v_ss,a): Gaussian with tempered 2×2 covariance.
    ``d_tilde``: mixture CDF for PIT; symmetric Gaussian-style bands using
    moment-matched marginal mean/variance (same z-quantiles as the full Gaussian path).
    Block-diagonal implied correlation: cross-covariance between (v_ss,a) and d set to 0.
    """
    from .models_u import mixture_cdf_1d, mixture_marginal_mean_var

    model.eval()
    n = x_num.shape[0]
    tm = target_means.view(1, 3)
    ts = target_stds.view(1, 3)

    sum_w_nll = 0.0
    sum_u_nll = 0.0
    sum_w = 0.0
    sum_abs_err = torch.zeros(3, device=device)
    sum_sq_err = torch.zeros(3, device=device)
    cnt_cov50 = torch.zeros(3, device=device)
    cnt_cov80 = torch.zeros(3, device=device)
    cnt_cov90 = torch.zeros(3, device=device)
    sum_width50 = torch.zeros(3, device=device)
    sum_width80 = torch.zeros(3, device=device)
    sum_width90 = torch.zeros(3, device=device)
    sum_pred_var_raw = torch.zeros(3, device=device)
    sum_corr_mat = torch.zeros(3, 3, device=device)
    pits_store: list[np.ndarray] = []

    z50 = float(stats.norm.ppf(0.75))
    z80 = float(stats.norm.ppf(0.9))
    z90 = float(stats.norm.ppf(0.95))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        xb = x_num[sl]
        cb = {k: v[sl] for k, v in cat.items()}
        mu_va, tril_va, mix_logits, mix_mu, mix_scale = model(xb, cb)
        Ls = tril_va * float(np.sqrt(T_va))
        sig_mix = mix_scale * float(np.sqrt(T_mix))

        dist2 = torch.distributions.MultivariateNormal(mu_va, scale_tril=Ls, validate_args=False)
        lp2 = dist2.log_prob(y_std[sl, :2])
        from .models_u import mixture_log_prob_1d

        lpd = mixture_log_prob_1d(y_std[sl, 2], mix_logits, mix_mu, sig_mix)
        lp = lp2 + lpd
        nll = -lp
        wb = w[sl]
        sum_w_nll += float((nll * wb).sum().item())
        sum_u_nll += float(nll.sum().item())
        sum_w += float(wb.sum().item())

        Sig2 = torch.bmm(Ls, Ls.transpose(-1, -2))
        var_va = torch.diagonal(Sig2, dim1=-2, dim2=-1).clamp_min(1e-12)
        std_va = torch.sqrt(var_va)
        mmean_d, mvar_d = mixture_marginal_mean_var(mix_logits, mix_mu, sig_mix)
        std_d = torch.sqrt(mvar_d.clamp_min(1e-12))

        # Pointwise predictive mean (raw): (v_ss,a) from μ_va; d from mixture moment mean
        mu_raw_va = mu_va * ts[0, :2] + tm[0, :2]
        mu_raw_d = mmean_d * ts[0, 2] + tm[0, 2]
        mu_raw = torch.cat([mu_raw_va, mu_raw_d.unsqueeze(-1)], dim=-1)
        yb = y_raw[sl]
        err = yb - mu_raw
        sum_abs_err += torch.sum(torch.abs(err), dim=0)
        sum_sq_err += torch.sum(err**2, dim=0)

        # Coverage / widths for v_ss, a (Gaussian marginals)
        for k in range(2):
            mu_k = mu_va[:, k]
            sm = std_va[:, k]
            low50 = (mu_k - z50 * sm) * ts[0, k] + tm[0, k]
            high50 = (mu_k + z50 * sm) * ts[0, k] + tm[0, k]
            low80 = (mu_k - z80 * sm) * ts[0, k] + tm[0, k]
            high80 = (mu_k + z80 * sm) * ts[0, k] + tm[0, k]
            low90 = (mu_k - z90 * sm) * ts[0, k] + tm[0, k]
            high90 = (mu_k + z90 * sm) * ts[0, k] + tm[0, k]
            yk = yb[:, k]
            cnt_cov50[k] += float(((yk >= low50) & (yk <= high50)).sum().item())
            cnt_cov80[k] += float(((yk >= low80) & (yk <= high80)).sum().item())
            cnt_cov90[k] += float(((yk >= low90) & (yk <= high90)).sum().item())
            sum_width50[k] += float((high50 - low50).sum().item())
            sum_width80[k] += float((high80 - low80).sum().item())
            sum_width90[k] += float((high90 - low90).sum().item())

        sum_pred_var_raw[:2] += torch.sum(var_va * (ts[0, :2] ** 2), dim=0)

        inv_std0 = 1.0 / std_va[:, 0].clamp_min(1e-8)
        inv_std1 = 1.0 / std_va[:, 1].clamp_min(1e-8)
        pit0 = torch.special.ndtr((y_std[sl, 0] - mu_va[:, 0]) * inv_std0)
        pit1 = torch.special.ndtr((y_std[sl, 1] - mu_va[:, 1]) * inv_std1)
        pit2 = mixture_cdf_1d(y_std[sl, 2], mix_logits, mix_mu, sig_mix).clamp(0.0, 1.0)

        k = 2
        mu_d = mmean_d
        sm = std_d
        low50 = (mu_d - z50 * sm) * ts[0, 2] + tm[0, 2]
        high50 = (mu_d + z50 * sm) * ts[0, 2] + tm[0, 2]
        low80 = (mu_d - z80 * sm) * ts[0, 2] + tm[0, 2]
        high80 = (mu_d + z80 * sm) * ts[0, 2] + tm[0, 2]
        low90 = (mu_d - z90 * sm) * ts[0, 2] + tm[0, 2]
        high90 = (mu_d + z90 * sm) * ts[0, 2] + tm[0, 2]
        yk = yb[:, 2]
        cnt_cov50[k] += float(((yk >= low50) & (yk <= high50)).sum().item())
        cnt_cov80[k] += float(((yk >= low80) & (yk <= high80)).sum().item())
        cnt_cov90[k] += float(((yk >= low90) & (yk <= high90)).sum().item())
        sum_width50[k] += float((high50 - low50).sum().item())
        sum_width80[k] += float((high80 - low80).sum().item())
        sum_width90[k] += float((high90 - low90).sum().item())

        sum_pred_var_raw[2] += torch.sum(mvar_d * (ts[0, 2] ** 2))

        pit = torch.stack([pit0, pit1, pit2], dim=-1).cpu().numpy()
        pits_store.append(pit)

        std_rows = torch.sqrt(torch.diagonal(Sig2, dim1=-2, dim2=-1).clamp_min(1e-12))
        corr_b2 = Sig2 / (std_rows[:, :, None] * std_rows[:, None, :] + 1e-12)
        sum_corr_mat[:2, :2] += torch.sum(corr_b2, dim=0)

    inv_total = 1.0 / float(max(n, 1))
    wmean_nll = sum_w_nll / max(sum_w, 1e-12)
    umean_nll = sum_u_nll / max(n, 1)
    mae = (sum_abs_err * inv_total).cpu().numpy().tolist()
    rmse = torch.sqrt(sum_sq_err * inv_total).cpu().numpy().tolist()
    cov50 = (cnt_cov50 * inv_total).cpu().numpy().tolist()
    cov80 = (cnt_cov80 * inv_total).cpu().numpy().tolist()
    cov90 = (cnt_cov90 * inv_total).cpu().numpy().tolist()
    width50 = (sum_width50 * inv_total).cpu().numpy().tolist()
    width80 = (sum_width80 * inv_total).cpu().numpy().tolist()
    width90 = (sum_width90 * inv_total).cpu().numpy().tolist()
    avg_pred_std_raw = torch.sqrt(sum_pred_var_raw * inv_total).cpu().numpy().tolist()
    pits = np.concatenate(pits_store, axis=0)
    empirical_std_raw = np.std(y_raw.cpu().numpy(), axis=0).tolist()
    hist_counts = [np.histogram(pits[:, k], bins=20, range=(0, 1))[0].tolist() for k in range(3)]

    mean_corr = torch.zeros(3, 3, device=device)
    mean_corr[:2, :2] = sum_corr_mat[:2, :2] * inv_total
    mean_corr[2, 2] = 1.0
    mean_corr[0, 2] = mean_corr[2, 0] = 0.0
    mean_corr[1, 2] = mean_corr[2, 1] = 0.0

    return {
        "n_rows": int(n),
        "weighted_nll": wmean_nll,
        "unweighted_nll": umean_nll,
        "mae_raw": mae,
        "rmse_raw": rmse,
        "target_names": list(U_NAMES),
        "coverage_50_marginal": cov50,
        "coverage_80_marginal": cov80,
        "coverage_90_marginal": cov90,
        "interval_width_50_raw_mean": width50,
        "interval_width_80_raw_mean": width80,
        "interval_width_90_raw_mean": width90,
        "avg_pred_std_marginal_raw": avg_pred_std_raw,
        "empirical_std_raw": empirical_std_raw,
        "mean_predicted_corr": mean_corr.cpu().numpy().tolist(),
        "pit_hist_counts": hist_counts,
        "eval_note": (
            "Hybrid model: d_tilde PIT from exact mixture CDF; d intervals use moment-matched Gaussian "
            "symmetric bands; corr cross-terms (v,a) vs d set to 0."
        ),
    }


def _wrap_deg_delta(delta_deg: np.ndarray) -> np.ndarray:
    """Wrap difference to (-180, 180]."""
    return (delta_deg + 180.0) % 360.0 - 180.0


def _mixture_marginal_cdf_ndtr(
    y_std: torch.Tensor,
    mix_logits: torch.Tensor,
    mu_blk: torch.Tensor,
    L_blk: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """1D marginal CDF of Σ_k π_k N(μ_k, L_k L_k^T) at coordinate ``dim``; y_std (B,)."""
    pi = torch.softmax(mix_logits, dim=-1)
    Sig = torch.matmul(L_blk, L_blk.transpose(-1, -2))
    sig = torch.sqrt(torch.diagonal(Sig, dim1=-2, dim2=-1).clamp_min(1e-12)[..., dim])
    mu_m = mu_blk[..., dim]
    z = (y_std.unsqueeze(1) - mu_m) / sig.clamp_min(1e-8)
    terms = pi * torch.special.ndtr(z)
    return terms.sum(dim=-1).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_shared_gaussian_vm_split_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_std: torch.Tensor,
    y_raw: torch.Tensor,
    w: torch.Tensor,
    target_means: torch.Tensor,
    target_stds: torch.Tensor,
    *,
    T_va: float = 1.0,
    T_ang: float = 1.0,
    batch_size: int = 4096,
    device: torch.device,
    n_mc_circle: int = 256,
    kappa_max: float = 120.0,
    return_pit_array: bool = False,
    va_transform: Any | None = None,
    player_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Metrics for SharedMixtureGaussianVonMisesUNet: Gaussian marginals (v_ss,a) from
    mixture; **d_tilde** with von Mises mixture CDF PIT (scipy), wrapped MAE/RMSE, MC central 50% coverage.
    """
    from scipy.stats import vonmises

    from .models_u import SharedMixtureGaussianVonMisesUNet

    if not isinstance(model, SharedMixtureGaussianVonMisesUNet):
        pass  # duck typing ok

    model.eval()
    n = x_num.shape[0]
    tm = target_means.view(1, 3)
    ts = target_stds.view(1, 3)

    sum_w_nll = 0.0
    sum_u_nll = 0.0
    sum_w = 0.0
    sum_abs_err = torch.zeros(3, device=device)
    sum_sq_err = torch.zeros(3, device=device)
    cnt_cov50 = torch.zeros(3, device=device)
    cnt_cov80 = torch.zeros(3, device=device)
    cnt_cov90 = torch.zeros(3, device=device)
    sum_width50 = torch.zeros(3, device=device)
    sum_width80 = torch.zeros(3, device=device)
    sum_width90 = torch.zeros(3, device=device)
    sum_pred_var_raw = torch.zeros(3, device=device)
    sum_corr_mat = torch.zeros(3, 3, device=device)
    pits_store: list[np.ndarray] = []

    z50 = float(stats.norm.ppf(0.75))
    z80 = float(stats.norm.ppf(0.9))
    z90 = float(stats.norm.ppf(0.95))

    rng = np.random.default_rng(20260408)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        xb = x_num[sl]
        cb = {k: v[sl] for k, v in cat.items()}
        y_s = y_std[sl]
        yb = y_raw[sl]
        wb = w[sl]
        bid_np = player_ids[sl].detach().cpu().numpy() if player_ids is not None else None

        logits, mu_va, L_va, loc_d, kappa_d = model(xb, cb)
        Ls = L_va * float(np.sqrt(T_va))
        kc = (kappa_d / float(T_ang)).clamp(min=1e-4, max=float(kappa_max))

        from .models_u import shared_mixture_bivariate_gaussian_vonmises_log_prob

        lp = shared_mixture_bivariate_gaussian_vonmises_log_prob(
            y_s[:, :2],
            torch.deg2rad(yb[:, 2]),
            logits,
            mu_va,
            Ls,
            loc_d,
            kc,
            kappa_max=kappa_max,
        )
        nll = -lp
        sum_w_nll += float((nll * wb).sum().item())
        sum_u_nll += float(nll.sum().item())
        sum_w += float(wb.sum().item())

        B = y_s.shape[0]
        K = logits.shape[1]
        pi = torch.softmax(logits, dim=-1).cpu().numpy()
        loc_np = loc_d.detach().cpu().numpy()
        kap_np = kc.detach().cpu().numpy()
        d_obs_rad = np.deg2rad(yb[:, 2].detach().cpu().numpy())

        mu_pred_deg = np.degrees(
            np.arctan2(
                np.sum(pi * np.sin(loc_np), axis=1),
                np.sum(pi * np.cos(loc_np), axis=1),
            )
        )
        ydeg = yb[:, 2].detach().cpu().numpy()
        cdf_k = vonmises.cdf(d_obs_rad[:, np.newaxis], kap_np, loc=loc_np)
        pit2 = np.clip(np.sum(pi * cdf_k, axis=1), 0.0, 1.0).astype(np.float64)

        # Mixture samples via Gumbel–max (same categorical draws as multinomial); batched rvs.
        log_pi = np.log(np.maximum(pi, 1e-20))
        gumbel = rng.gumbel(size=(B, n_mc_circle, K))
        ks = np.argmax(log_pi[:, None, :] + gumbel, axis=-1)
        br = np.arange(B, dtype=np.int64)[:, None]
        kap_s = kap_np[br, ks]
        loc_s = loc_np[br, ks]
        samps = vonmises.rvs(kap_s, loc=loc_s)
        s_deg = np.degrees(samps)
        delta = _wrap_deg_delta(s_deg - ydeg[:, None])
        q25 = np.quantile(delta, 0.25, axis=1)
        q75 = np.quantile(delta, 0.75, axis=1)
        cov50_d = int(np.sum((q25 <= 0.0) & (0.0 <= q75)))
        ed = _wrap_deg_delta(mu_pred_deg - ydeg)
        sum_abs_d_acc = float(np.sum(np.abs(ed)))
        sum_sq_d_acc = float(np.sum(ed * ed))

        cnt_cov50[2] += float(cov50_d)
        sum_abs_err[2] += sum_abs_d_acc
        sum_sq_err[2] += sum_sq_d_acc

        pi_t = torch.softmax(logits, dim=-1).unsqueeze(-1)
        mu_std_0 = (pi_t * mu_va[:, :, 0:1]).sum(dim=1).squeeze(-1)
        mu_std_1 = (pi_t * mu_va[:, :, 1:2]).sum(dim=1).squeeze(-1)
        if va_transform is not None:
            mu_va_raw_np = _inverse_va_transform(
                va_transform,
                torch.stack([mu_std_0, mu_std_1], dim=1).detach().cpu().numpy(),
                bid_np,
            )
            mu_raw_0 = torch.as_tensor(mu_va_raw_np[:, 0], device=device, dtype=yb.dtype)
            mu_raw_1 = torch.as_tensor(mu_va_raw_np[:, 1], device=device, dtype=yb.dtype)
        else:
            mu_raw_0 = mu_std_0 * ts[0, 0] + tm[0, 0]
            mu_raw_1 = mu_std_1 * ts[0, 1] + tm[0, 1]
        mu_pred_t = torch.from_numpy(mu_pred_deg.astype(np.float32)).to(device=device)
        mu_raw = torch.stack([mu_raw_0, mu_raw_1, mu_pred_t], dim=-1)
        err = yb - mu_raw
        sum_abs_err[0] += torch.sum(torch.abs(err[:, 0]))
        sum_abs_err[1] += torch.sum(torch.abs(err[:, 1]))
        sum_sq_err[0] += torch.sum(err[:, 0] ** 2)
        sum_sq_err[1] += torch.sum(err[:, 1] ** 2)

        pit0 = _mixture_marginal_cdf_ndtr(y_s[:, 0], logits, mu_va, Ls, dim=0)
        pit1 = _mixture_marginal_cdf_ndtr(y_s[:, 1], logits, mu_va, Ls, dim=1)

        Sig2_k = torch.matmul(Ls, Ls.transpose(-1, -2))
        var_m0 = (
            torch.sum(pi_t.squeeze(-1) * Sig2_k[:, :, 0, 0], dim=1)
            + torch.sum(pi_t.squeeze(-1) * (mu_va[:, :, 0] ** 2), dim=1)
            - mu_std_0**2
        )
        var_m1 = (
            torch.sum(pi_t.squeeze(-1) * Sig2_k[:, :, 1, 1], dim=1)
            + torch.sum(pi_t.squeeze(-1) * (mu_va[:, :, 1] ** 2), dim=1)
            - mu_std_1**2
        )
        std_m0 = torch.sqrt(var_m0.clamp_min(1e-12))
        std_m1 = torch.sqrt(var_m1.clamp_min(1e-12))


        for kdim, mu_s, sm, tk in [
            (0, mu_std_0, std_m0, 0),
            (1, mu_std_1, std_m1, 1),
        ]:
            if va_transform is not None:
                zeros = np.zeros((mu_s.shape[0], 2), dtype=np.float32)
                def _inv_dim(vals: torch.Tensor) -> torch.Tensor:
                    arr = zeros.copy()
                    arr[:, kdim] = vals.detach().cpu().numpy()
                    other = mu_std_1 if kdim == 0 else mu_std_0
                    arr[:, 1 - kdim] = other.detach().cpu().numpy()
                    raw = _inverse_va_transform(va_transform, arr, bid_np)
                    return torch.as_tensor(raw[:, kdim], device=device, dtype=yb.dtype)

                low50 = _inv_dim(mu_s - z50 * sm)
                high50 = _inv_dim(mu_s + z50 * sm)
                low80 = _inv_dim(mu_s - z80 * sm)
                high80 = _inv_dim(mu_s + z80 * sm)
                low90 = _inv_dim(mu_s - z90 * sm)
                high90 = _inv_dim(mu_s + z90 * sm)
            else:
                low50 = (mu_s - z50 * sm) * ts[0, tk] + tm[0, tk]
                high50 = (mu_s + z50 * sm) * ts[0, tk] + tm[0, tk]
                low80 = (mu_s - z80 * sm) * ts[0, tk] + tm[0, tk]
                high80 = (mu_s + z80 * sm) * ts[0, tk] + tm[0, tk]
                low90 = (mu_s - z90 * sm) * ts[0, tk] + tm[0, tk]
                high90 = (mu_s + z90 * sm) * ts[0, tk] + tm[0, tk]
            yk = yb[:, kdim]
            cnt_cov50[kdim] += float(((yk >= low50) & (yk <= high50)).sum().item())
            cnt_cov80[kdim] += float(((yk >= low80) & (yk <= high80)).sum().item())
            cnt_cov90[kdim] += float(((yk >= low90) & (yk <= high90)).sum().item())
            sum_width50[kdim] += float((high50 - low50).sum().item())
            sum_width80[kdim] += float((high80 - low80).sum().item())
            sum_width90[kdim] += float((high90 - low90).sum().item())

        sum_pred_var_raw[0] += torch.sum(var_m0 * (ts[0, 0] ** 2))
        sum_pred_var_raw[1] += torch.sum(var_m1 * (ts[0, 1] ** 2))

        inv_k = 1.0 / kc.clamp_min(1e-4)
        e_inv_k = (pi_t.squeeze(-1) * inv_k).sum(dim=1)
        sum_pred_var_raw[2] += torch.sum(e_inv_k * (180.0 / np.pi) ** 2)

        pit = np.stack(
            [pit0.detach().cpu().numpy(), pit1.detach().cpu().numpy(), pit2],
            axis=1,
        )
        pits_store.append(pit)

        # Corr from weighted average component correlation + between-mean term (approx)
        std_rows = torch.sqrt(torch.diagonal(Sig2_k, dim1=-2, dim2=-1).clamp_min(1e-12))
        # Outer product per (batch, k): [B,K,2,1]*[B,K,1,2]; NOT [:,None,:] which mixes K into wrong axis.
        corr_k = Sig2_k / (
            std_rows.unsqueeze(-1) * std_rows.unsqueeze(-2) + 1e-12
        )
        w_k = pi_t.squeeze(-1)  # [B, K]; avoid pi_t[..., None, None] which is 5-D on [B,K,1]
        wavg_corr = (w_k[:, :, None, None] * corr_k).sum(dim=1)
        sum_corr_mat[:2, :2] += torch.sum(wavg_corr, dim=0)

    inv_total = 1.0 / float(max(n, 1))
    wmean_nll = sum_w_nll / max(sum_w, 1e-12)
    umean_nll = sum_u_nll / max(n, 1)
    mae = (sum_abs_err * inv_total).cpu().numpy().tolist()
    rmse = torch.sqrt(sum_sq_err * inv_total).cpu().numpy().tolist()
    cov50 = (cnt_cov50 * inv_total).cpu().numpy().tolist()
    cov80 = (cnt_cov80 * inv_total).cpu().numpy().tolist()
    cov90 = (cnt_cov90 * inv_total).cpu().numpy().tolist()
    width50 = (sum_width50 * inv_total).cpu().numpy().tolist()
    width80 = (sum_width80 * inv_total).cpu().numpy().tolist()
    width90 = (sum_width90 * inv_total).cpu().numpy().tolist()
    cov80[2] = float("nan")
    cov90[2] = float("nan")
    width50[2] = float("nan")
    width80[2] = float("nan")
    width90[2] = float("nan")
    avg_pred_std_raw = torch.sqrt(sum_pred_var_raw * inv_total).cpu().numpy().tolist()
    pits = np.concatenate(pits_store, axis=0)
    empirical_std_raw = np.std(y_raw.cpu().numpy(), axis=0).tolist()
    hist_counts = [np.histogram(pits[:, k], bins=20, range=(0, 1))[0].tolist() for k in range(3)]

    mean_corr = torch.zeros(3, 3, device=device)
    mean_corr[:2, :2] = sum_corr_mat[:2, :2] * inv_total
    mean_corr[2, 2] = 1.0

    return {
        "n_rows": int(n),
        "weighted_nll": wmean_nll,
        "unweighted_nll": umean_nll,
        "mae_raw": mae,
        "rmse_raw": rmse,
        "target_names": list(U_NAMES),
        "coverage_50_marginal": cov50,
        "coverage_80_marginal": cov80,
        "coverage_90_marginal": cov90,
        "interval_width_50_raw_mean": width50,
        "interval_width_80_raw_mean": width80,
        "interval_width_90_raw_mean": width90,
        "avg_pred_std_marginal_raw": avg_pred_std_raw,
        "empirical_std_raw": empirical_std_raw,
        "mean_predicted_corr": mean_corr.cpu().numpy().tolist(),
        "pit_hist_counts": hist_counts,
        "d_tilde_mc_samples_for_coverage": int(n_mc_circle),
        "eval_note": (
            "Shared π_k: (v_ss,a) mixture Gaussian marginals; d_tilde von Mises mixture; "
            "d PIT via scipy vonmises.cdf; d MAE uses wrapped deg error vs mixture circular mean; "
            "d nominal 50% via MC central interval on wrapped samples; "
            "d marginal 80/90 coverage and widths omitted (NaN). "
            "mean_predicted_corr[:2,:2] is a π_k-weighted average of within-component correlations only "
            "(no v/a–d cross terms reported)."
        ),
        **({"pit_values": pits} if return_pit_array else {}),
    }


@torch.no_grad()
def _inverse_va_transform(
    va_transform: Any,
    ytilde: np.ndarray,
    player_ids: np.ndarray | None,
) -> np.ndarray:
    from .target_transforms_u import PlayerSpecificBoundedLogitZScoreTransform

    if isinstance(va_transform, PlayerSpecificBoundedLogitZScoreTransform):
        if player_ids is None:
            raise ValueError("player_ids required for player-specific transform")
        return va_transform.inverse(ytilde, player_ids)
    return va_transform.inverse(ytilde)


def evaluate_bounded_hier_shared_gaussian_vm_split_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    batter_id: torch.Tensor,
    y_model: torch.Tensor,
    y_raw: torch.Tensor,
    w: torch.Tensor,
    va_transform: Any,
    *,
    T_va: float = 1.0,
    T_ang: float = 1.0,
    batch_size: int = 4096,
    device: torch.device,
    n_mc_circle: int = 256,
    kappa_max: float = 120.0,
    return_pit_array: bool = False,
    report_player_diagnostics: bool = True,
    report_mixture_diagnostics: bool = True,
    report_covariance_diagnostics: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Metrics for BoundedHierSharedMixtureGaussianVonMisesUNet."""
    from scipy.stats import vonmises

    from .models_u import BoundedHierSharedMixtureGaussianVonMisesUNet
    from .target_transforms_u import (
        BoundedLogitZScoreTransform,
        PlayerSpecificBoundedLogitZScoreTransform,
    )

    if not isinstance(
        va_transform, (BoundedLogitZScoreTransform, PlayerSpecificBoundedLogitZScoreTransform)
    ):
        raise TypeError("va_transform must be a bounded logit transform")

    cat_ord = {k: v for k, v in cat.items() if k != "batter_name"}
    model.eval()
    n = x_num.shape[0]

    sum_w_nll_model = 0.0
    sum_w_nll_raw = 0.0
    sum_w = 0.0
    sum_abs_err = torch.zeros(3, device=device)
    sum_sq_err = torch.zeros(3, device=device)
    cnt_cov50 = torch.zeros(3, device=device)
    pits_store: list[np.ndarray] = []
    k_eff_all: list[np.ndarray] = []
    ent_all: list[np.ndarray] = []
    comp_usage = None
    diag_floor = float(getattr(model, "diag_floor", 0.05))
    diag_ceil = float(getattr(model, "diag_ceiling", 5.0))
    floor_hits = 0
    ceil_hits = 0
    n_diag = 0
    offdiag_sq_sum = 0.0
    n_off = 0
    per_player_nll: dict[int, list[float]] = {}
    per_player_w: dict[int, list[float]] = {}

    z50 = float(stats.norm.ppf(0.75))
    rng = np.random.default_rng(20260408)
    K = getattr(model, "n_mixture", 6)
    if comp_usage is None:
        comp_usage = np.zeros(K, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        xb = x_num[sl]
        cb = {k: v[sl] for k, v in cat_ord.items()}
        bbid = batter_id[sl]
        y_m = y_model[sl]
        yb = y_raw[sl]
        wb = w[sl]
        bid_np = bbid.detach().cpu().numpy()

        logits, mu_va, L_va, loc_d, kappa_d = model(xb, cb, bbid)
        Ls = L_va * float(np.sqrt(T_va))
        kc = (kappa_d / float(T_ang)).clamp(min=1e-4, max=float(kappa_max))

        from .models_u import shared_mixture_bivariate_gaussian_vonmises_log_prob

        lp = shared_mixture_bivariate_gaussian_vonmises_log_prob(
            y_m[:, :2],
            torch.deg2rad(yb[:, 2]),
            logits,
            mu_va,
            Ls,
            loc_d,
            kc,
            kappa_max=kappa_max,
        )
        nll = -lp
        sum_w_nll_model += float((nll * wb).sum().item())
        sum_w += float(wb.sum().item())

        raw_va_np = yb[:, :2].detach().cpu().numpy()
        if isinstance(va_transform, PlayerSpecificBoundedLogitZScoreTransform):
            log_jac = va_transform.log_abs_det_dytilde_dy(raw_va_np, bid_np)
        else:
            log_jac = va_transform.log_abs_det_dytilde_dy(raw_va_np)
        sum_w_nll_raw += float((nll * wb).sum().item()) - float((torch.from_numpy(log_jac).to(device) * wb).sum().item())

        pi = torch.softmax(logits, dim=-1).detach()
        if report_mixture_diagnostics:
            k_eff_all.append(
                BoundedHierSharedMixtureGaussianVonMisesUNet.mixture_effective_components(pi).cpu().numpy()
            )
            ent_all.append(
                BoundedHierSharedMixtureGaussianVonMisesUNet.mixture_entropy(pi).cpu().numpy()
            )
            comp_usage += pi.sum(dim=0).cpu().numpy()

        if report_covariance_diagnostics and hasattr(model, "_last_chol_diag"):
            diag = torch.exp(model._last_chol_diag).detach().cpu().numpy()
            floor_hits += int(np.sum(diag <= diag_floor * 1.01))
            ceil_hits += int(np.sum(diag >= diag_ceil * 0.99))
            n_diag += diag.size
            off = model._last_chol_offdiag.detach().cpu().numpy()
            offdiag_sq_sum += float((off**2).sum())
            n_off += off.size

        B = y_m.shape[0]
        pi_np = pi.cpu().numpy()
        loc_np = loc_d.detach().cpu().numpy()
        kap_np = kc.detach().cpu().numpy()
        d_obs_rad = np.deg2rad(yb[:, 2].detach().cpu().numpy())

        mu_pred_deg = np.degrees(
            np.arctan2(
                np.sum(pi_np * np.sin(loc_np), axis=1),
                np.sum(pi_np * np.cos(loc_np), axis=1),
            )
        )
        ydeg = yb[:, 2].detach().cpu().numpy()
        cdf_k = vonmises.cdf(d_obs_rad[:, np.newaxis], kap_np, loc=loc_np)
        pit2 = np.clip(np.sum(pi_np * cdf_k, axis=1), 0.0, 1.0)

        log_pi = np.log(np.maximum(pi_np, 1e-20))
        gumbel = rng.gumbel(size=(B, n_mc_circle, pi_np.shape[1]))
        ks = np.argmax(log_pi[:, None, :] + gumbel, axis=-1)
        br = np.arange(B, dtype=np.int64)[:, None]
        samps = vonmises.rvs(kap_np[br, ks], loc=loc_np[br, ks])
        delta = _wrap_deg_delta(np.degrees(samps) - ydeg[:, None])
        q25 = np.quantile(delta, 0.25, axis=1)
        q75 = np.quantile(delta, 0.75, axis=1)
        cnt_cov50[2] += float(np.sum((q25 <= 0.0) & (0.0 <= q75)))

        pit0 = _mixture_marginal_cdf_ndtr(y_m[:, 0], logits, mu_va, Ls, dim=0)
        pit1 = _mixture_marginal_cdf_ndtr(y_m[:, 1], logits, mu_va, Ls, dim=1)
        pits_store.append(
            np.stack(
                [pit0.detach().cpu().numpy(), pit1.detach().cpu().numpy(), pit2],
                axis=1,
            )
        )

        pi_t = pi.unsqueeze(-1)
        mu_std_0 = (pi_t * mu_va[:, :, 0:1]).sum(dim=1).squeeze(-1)
        mu_std_1 = (pi_t * mu_va[:, :, 1:2]).sum(dim=1).squeeze(-1)
        yt_stack = torch.stack([mu_std_0, mu_std_1], dim=-1).detach().cpu().numpy()
        mu_raw_va = _inverse_va_transform(va_transform, yt_stack, bid_np)
        mu_raw = np.column_stack([mu_raw_va, mu_pred_deg.astype(np.float32)])
        err_np = yb.detach().cpu().numpy() - mu_raw
        sum_abs_err += torch.from_numpy(np.abs(err_np).sum(axis=0)).to(device)
        sum_sq_err += torch.from_numpy((err_np**2).sum(axis=0)).to(device)

        Sig2_k = torch.matmul(Ls, Ls.transpose(-1, -2))
        var_m0 = (
            torch.sum(pi.squeeze(-1) * Sig2_k[:, :, 0, 0], dim=1)
            + torch.sum(pi.squeeze(-1) * (mu_va[:, :, 0] ** 2), dim=1)
            - mu_std_0**2
        )
        var_m1 = (
            torch.sum(pi.squeeze(-1) * Sig2_k[:, :, 1, 1], dim=1)
            + torch.sum(pi.squeeze(-1) * (mu_va[:, :, 1] ** 2), dim=1)
            - mu_std_1**2
        )
        std_m0 = torch.sqrt(var_m0.clamp_min(1e-12))
        std_m1 = torch.sqrt(var_m1.clamp_min(1e-12))
        yt_mu = torch.from_numpy(mu_raw_va).to(device=device, dtype=y_m.dtype)
        yt_std0 = torch.sqrt(var_m0.clamp_min(1e-12))
        yt_std1 = torch.sqrt(var_m1.clamp_min(1e-12))
        low50_0 = mu_std_0 - z50 * yt_std0
        high50_0 = mu_std_0 + z50 * yt_std0
        low50_1 = mu_std_1 - z50 * yt_std1
        high50_1 = mu_std_1 + z50 * yt_std1
        yt_lo = np.column_stack([low50_0.detach().cpu().numpy(), low50_1.detach().cpu().numpy()])
        yt_hi = np.column_stack([high50_0.detach().cpu().numpy(), high50_1.detach().cpu().numpy()])
        raw_lo = _inverse_va_transform(va_transform, yt_lo, bid_np)
        raw_hi = _inverse_va_transform(va_transform, yt_hi, bid_np)
        raw_low0, raw_low1 = raw_lo[:, 0], raw_lo[:, 1]
        raw_high0, raw_high1 = raw_hi[:, 0], raw_hi[:, 1]
        y0 = yb[:, 0].detach().cpu().numpy()
        y1 = yb[:, 1].detach().cpu().numpy()
        cnt_cov50[0] += float(np.sum((y0 >= raw_low0) & (y0 <= raw_high0)))
        cnt_cov50[1] += float(np.sum((y1 >= raw_low1) & (y1 <= raw_high1)))

        if report_player_diagnostics:
            for i in range(B):
                pid = int(bbid[i].item())
                per_player_nll.setdefault(pid, []).append(float(nll[i].item()))
                per_player_w.setdefault(pid, []).append(float(wb[i].item()))

    inv_total = 1.0 / float(max(n, 1))
    wmean_model = sum_w_nll_model / max(sum_w, 1e-12)
    wmean_raw = sum_w_nll_raw / max(sum_w, 1e-12)
    mae = (sum_abs_err * inv_total).cpu().numpy().tolist()
    rmse = torch.sqrt(sum_sq_err * inv_total).cpu().numpy().tolist()
    cov50 = (cnt_cov50 * inv_total).cpu().numpy().tolist()
    pits = np.concatenate(pits_store, axis=0)

    pack: dict[str, Any] = {
        "n_rows": int(n),
        "weighted_nll": wmean_model,
        "model_space_weighted_nll": wmean_model,
        "raw_space_weighted_nll_with_jacobian": wmean_raw,
        "mae_raw": mae,
        "rmse_raw": rmse,
        "target_names": list(U_NAMES),
        "coverage_50_marginal": cov50,
        "head_family": BoundedHierSharedMixtureGaussianVonMisesUNet.HEAD_FAMILY,
        "eval_note": "Bounded hier: va in logit-z space; raw metrics via inverse transform.",
    }

    if bounds:
        yraw_np = y_raw.cpu().numpy()
        viol = {}
        for j, name in enumerate(("v_ss_tilde", "a_tilde")):
            L, U = bounds[name]
            v = yraw_np[:, j]
            viol[name] = float(np.mean((v < L) | (v > U)))
        pack["physical_bound_violation_rate_after_sampling"] = viol

    if report_mixture_diagnostics and k_eff_all:
        ke = np.concatenate(k_eff_all)
        ent = np.concatenate(ent_all)
        pack["mixture_diagnostics"] = {
            "K_eff_mean": float(ke.mean()),
            "K_eff_median": float(np.median(ke)),
            "K_eff_p5": float(np.quantile(ke, 0.05)),
            "K_eff_p95": float(np.quantile(ke, 0.95)),
            "mixture_entropy_mean": float(ent.mean()),
            "component_usage_mean": (comp_usage / max(n, 1)).tolist(),
        }

    if report_covariance_diagnostics and n_diag > 0:
        pack["covariance_diagnostics"] = {
            "diag_near_floor_pct": 100.0 * floor_hits / n_diag,
            "diag_near_ceiling_pct": 100.0 * ceil_hits / n_diag,
            "offdiag_sq_mean": offdiag_sq_sum / max(n_off, 1),
        }

    if report_player_diagnostics and per_player_nll:
        rows = []
        for pid, nlls in per_player_nll.items():
            ws = np.array(per_player_w[pid])
            nll_a = np.array(nlls)
            wn = float((nll_a * ws).sum() / max(ws.sum(), 1e-12))
            rows.append({"batter_id": pid, "weighted_nll": wn, "n_draws": len(nlls)})
        rows.sort(key=lambda r: r["weighted_nll"])
        pack["player_diagnostics"] = {
            "worst_10_calibration_nll": rows[-10:][::-1],
            "best_10_calibration_nll": rows[:10],
            "n_players": len(rows),
        }

    pack["pit_hist_counts"] = [
        np.histogram(pits[:, k], bins=20, range=(0, 1))[0].tolist() for k in range(3)
    ]
    if return_pit_array:
        pack["pit_values"] = pits
    return pack


def save_player_embedding_norms_csv(
    model: torch.nn.Module,
    vocabs: dict[str, dict[str, int]],
    out_path: Path,
) -> None:
    import csv

    emb = model.player_embedding.weight.detach().cpu().numpy()
    inv = {v: k for k, v in vocabs["batter_name"].items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["batter_name", "batter_id", "embedding_l2_norm"])
        for i in range(emb.shape[0]):
            name = inv.get(i, f"id_{i}")
            w.writerow([name, i, float(np.linalg.norm(emb[i]))])


def save_metrics_json_csv(out_dir: Path, name: str, pack: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_out = {k: v for k, v in pack.items() if k != "pit_values"}
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(pack_out, indent=2), encoding="utf-8")
    import csv

    flat = {
        k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
        for k, v in pack.items()
        if k != "pit_values"
    }
    with (out_dir / f"{name}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(flat.keys())
        w.writerow(flat.values())


def plot_pit_hist(pits: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for k, ax in enumerate(axes):
        ax.hist(pits[:, k], bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax.set_title(U_NAMES[k])
        ax.set_xlabel("PIT")
    fig.suptitle("Marginal PIT (uniform if calibrated)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
