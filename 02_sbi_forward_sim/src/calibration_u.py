"""Covariance temperature scaling: Σ_cal = T Σ with scalar T > 0."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TemperatureScale:
    """Fitted scalar T; scale_tril_cal = sqrt(T) * scale_tril."""

    T: float

    def scale_tril(self, L: torch.Tensor) -> torch.Tensor:
        return L * math.sqrt(self.T)


def fit_global_covariance_temperature(
    mu: torch.Tensor,
    scale_tril: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    max_iter: int = 400,
    lr: float = 0.05,
) -> TemperatureScale:
    """
    Minimize weighted Gaussian NLL w.r.t. T>0 where calibrated Σ = T Σ.

    Uses Adam on log T for scalability (full calibration split in one pass).
    """
    log_t = torch.zeros(1, device=mu.device, dtype=mu.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    mu_d = mu.detach()
    L0 = scale_tril.detach()
    y_d = y.detach()
    w_d = w.detach()

    for _ in range(max_iter):
        opt.zero_grad()
        T = torch.exp(log_t).clamp(min=1e-8) + 1e-8
        L = L0 * torch.sqrt(T)
        dist = torch.distributions.MultivariateNormal(mu_d, scale_tril=L, validate_args=False)
        lp = dist.log_prob(y_d)
        nll = -lp
        loss = (nll * w_d).sum() / w_d.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()

    T_final = float((torch.exp(log_t).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    return TemperatureScale(T=T_final)


@dataclass
class HybridTemperatureScale:
    """T_va scales Σ for (v_ss,a); T_mix scales each mixture σ for d_tilde."""

    T_va: float
    T_mix: float


def fit_hybrid_va_gmm_temperatures(
    mu_va: torch.Tensor,
    scale_tril_va: torch.Tensor,
    mix_logits: torch.Tensor,
    mix_mu: torch.Tensor,
    mix_scale: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    max_iter: int = 400,
    lr: float = 0.05,
) -> HybridTemperatureScale:
    """
    Two scalar temperatures: L_cal = sqrt(T_va) L; σ_k,cal = sqrt(T_mix) σ_k.

    Minimizes weighted negative joint log p on the calibration split.
    """
    from .models_u import mixture_log_prob_1d

    log_t_va = torch.zeros(1, device=mu_va.device, dtype=mu_va.dtype, requires_grad=True)
    log_t_m = torch.zeros(1, device=mu_va.device, dtype=mu_va.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_t_va, log_t_m], lr=lr)

    mu_va_d = mu_va.detach()
    L0 = scale_tril_va.detach()
    ml = mix_logits.detach()
    mm = mix_mu.detach()
    ms = mix_scale.detach()
    y_d = y.detach()
    w_d = w.detach()

    for _ in range(max_iter):
        opt.zero_grad()
        t_va = torch.exp(log_t_va).clamp(min=1e-8) + 1e-8
        t_m = torch.exp(log_t_m).clamp(min=1e-8) + 1e-8
        L = L0 * torch.sqrt(t_va)
        dist = torch.distributions.MultivariateNormal(mu_va_d, scale_tril=L, validate_args=False)
        lp2 = dist.log_prob(y_d[:, :2])
        sig = ms * torch.sqrt(t_m)
        lpd = mixture_log_prob_1d(y_d[:, 2], ml, mm, sig)
        lp = lp2 + lpd
        nll = -lp
        loss = (nll * w_d).sum() / w_d.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()

    t_va_f = float((torch.exp(log_t_va).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    t_m_f = float((torch.exp(log_t_m).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    return HybridTemperatureScale(T_va=t_va_f, T_mix=t_m_f)


@dataclass
class HybridVaVmTemperatureScale:
    """T_va scales Σ for (v_ss,a); von Mises κ is divided by T_ang (wider angular spread if T_ang>1)."""

    T_va: float
    T_ang: float


def fit_hybrid_va_vm_temperatures(
    mu_va: torch.Tensor,
    scale_tril_va: torch.Tensor,
    mix_logits: torch.Tensor,
    loc_d: torch.Tensor,
    kappa_d: torch.Tensor,
    y: torch.Tensor,
    y_raw_d: torch.Tensor,
    w: torch.Tensor,
    *,
    max_iter: int = 400,
    lr: float = 0.05,
    kappa_max: float = 120.0,
) -> HybridVaVmTemperatureScale:
    """
    Scalar post-hoc calibration: L_cal = sqrt(T_va) L; κ_cal = κ / T_ang.

    Minimizes weighted negative joint log p on standardized (v_ss,a) and raw-degree d.
    """
    from .models_u import shared_mixture_bivariate_gaussian_vonmises_log_prob

    log_t_va = torch.zeros(1, device=mu_va.device, dtype=mu_va.dtype, requires_grad=True)
    log_t_ang = torch.zeros(1, device=mu_va.device, dtype=mu_va.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_t_va, log_t_ang], lr=lr)

    mu_d = mu_va.detach()
    L0 = scale_tril_va.detach()
    ml = mix_logits.detach()
    ld = loc_d.detach()
    kd = kappa_d.detach()
    y_d = y.detach()
    yrd = y_raw_d.detach()
    w_d = w.detach()

    for _ in range(max_iter):
        opt.zero_grad()
        t_va = torch.exp(log_t_va).clamp(min=1e-8) + 1e-8
        t_ang = torch.exp(log_t_ang).clamp(min=1e-8) + 1e-8
        L = L0 * torch.sqrt(t_va)
        kc = (kd / t_ang).clamp(min=1e-4, max=float(kappa_max))
        d_rad = torch.deg2rad(yrd)
        lp = shared_mixture_bivariate_gaussian_vonmises_log_prob(
            y_d[:, :2],
            d_rad,
            ml,
            mu_d,
            L,
            ld,
            kc,
            kappa_max=kappa_max,
        )
        nll = -lp
        loss = (nll * w_d).sum() / w_d.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()

    t_va_f = float((torch.exp(log_t_va).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    t_ang_f = float((torch.exp(log_t_ang).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    return HybridVaVmTemperatureScale(T_va=t_va_f, T_ang=t_ang_f)
