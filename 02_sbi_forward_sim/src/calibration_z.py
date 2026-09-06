"""Scalar T_mix on full-covariance mixture: Sigma_cal = T_mix * Sigma, i.e. L_cal = sqrt(T_mix) * L."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MDNTemperatureScale:
    T: float


def fit_mdn_temperature(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    max_iter: int = 400,
    lr: float = 0.05,
) -> MDNTemperatureScale:
    """Optimize scalar T_mix > 0; calibrated covariance T_mix * Sigma with L_cal = sqrt(T_mix) * L."""
    from .models_z import mdn_log_prob_chol

    log_t = torch.zeros(1, device=y.device, dtype=y.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    lg = mix_logits.detach()
    m = mu.detach()
    l0 = L.detach()
    y_d = y.detach()
    w_d = w.detach()

    for _ in range(max_iter):
        opt.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-8) + 1e-8
        scale = torch.sqrt(t)
        lcal = l0 * scale
        lp = mdn_log_prob_chol(y_d, lg, m, lcal)
        nll = -lp
        loss = (nll * w_d).sum() / w_d.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()

    t_f = float((torch.exp(log_t).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    return MDNTemperatureScale(T=t_f)


@dataclass
class HybridNativeTemperatureScale:
    """
    Row-wise Cholesky scaling: L_cal[i,:] *= sqrt(T_i) for i in {x, e_y*} (equivalent to Σ_cal = D Σ D, D=diag(√T)).
    κ_cal = κ / T_ang for both angles. Legacy scalar T_gauss equals T_gauss_x when isotropic (tx=ty).
    """

    T_ang: float
    T_gauss_x: float
    T_gauss_y: float

    @property
    def T_gauss(self) -> float:
        """Geometric mean; use old key `temperature_T_gauss` for coarse backward compatibility."""
        return float((self.T_gauss_x * self.T_gauss_y) ** 0.5)


def _hybrid_scaled_L(L_g: torch.Tensor, T_gauss_x: torch.Tensor, T_gauss_y: torch.Tensor) -> torch.Tensor:
    """L_g: (B,K,2,2); row i scaled by sqrt(T_i), matching Σ_cal = D Σ D with D = diag(√T_x, √T_y)."""
    s = torch.sqrt(torch.stack([T_gauss_x.reshape(()), T_gauss_y.reshape(())]).clamp(min=1e-8))
    row = s.view(1, 1, 2, 1).to(device=L_g.device, dtype=L_g.dtype)
    return L_g * row


def fit_hybrid_native_temperature(
    mix_logits: torch.Tensor,
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
    max_iter: int = 400,
    lr: float = 0.05,
    anisotropic_gaussian: bool = False,
) -> HybridNativeTemperatureScale:
    """Post-hoc calibration on calibration split: Gaussian cov inflation (isotropic or axis-wise); angular κ scale."""
    from .models_z import mdn_hybrid_gauss_vm_log_prob

    lg = mix_logits.detach()
    mg = mu_g.detach()
    l0 = L_g.detach()
    mp = mu_psi.detach()
    kp0 = kappa_psi.detach()
    mt = mu_theta.detach()
    kt0 = kappa_theta.detach()
    yd = y_g.detach()
    psid = psi.detach()
    thd = theta.detach()
    wd = w.detach()

    if anisotropic_gaussian:
        log_tx = torch.zeros(1, device=y_g.device, dtype=y_g.dtype, requires_grad=True)
        log_ty = torch.zeros(1, device=y_g.device, dtype=y_g.dtype, requires_grad=True)
        log_ta = torch.zeros(1, device=y_g.device, dtype=y_g.dtype, requires_grad=True)
        opt = torch.optim.Adam([log_tx, log_ty, log_ta], lr=lr)
        for _ in range(max_iter):
            opt.zero_grad()
            tx = torch.exp(log_tx).clamp(min=1e-8) + 1e-8
            ty = torch.exp(log_ty).clamp(min=1e-8) + 1e-8
            ta = torch.exp(log_ta).clamp(min=1e-8) + 1e-8
            lcal = _hybrid_scaled_L(l0, tx, ty)
            kpsc = kp0 / ta
            kthc = kt0 / ta
            lp = mdn_hybrid_gauss_vm_log_prob(yd, psid, thd, lg, mg, lcal, mp, kpsc, mt, kthc)
            nll = -lp
            loss = (nll * wd).sum() / wd.sum().clamp_min(1e-12)
            loss.backward()
            opt.step()
        tx_f = float((torch.exp(log_tx).clamp(min=1e-8) + 1e-8).detach().cpu().item())
        ty_f = float((torch.exp(log_ty).clamp(min=1e-8) + 1e-8).detach().cpu().item())
        ta_f = float((torch.exp(log_ta).clamp(min=1e-8) + 1e-8).detach().cpu().item())
        return HybridNativeTemperatureScale(T_ang=ta_f, T_gauss_x=tx_f, T_gauss_y=ty_f)

    log_tg = torch.zeros(1, device=y_g.device, dtype=y_g.dtype, requires_grad=True)
    log_ta = torch.zeros(1, device=y_g.device, dtype=y_g.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_tg, log_ta], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        tg = torch.exp(log_tg).clamp(min=1e-8) + 1e-8
        ta = torch.exp(log_ta).clamp(min=1e-8) + 1e-8
        lcal = l0 * torch.sqrt(tg)
        kpsc = kp0 / ta
        kthc = kt0 / ta
        lp = mdn_hybrid_gauss_vm_log_prob(yd, psid, thd, lg, mg, lcal, mp, kpsc, mt, kthc)
        nll = -lp
        loss = (nll * wd).sum() / wd.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()

    tg_f = float((torch.exp(log_tg).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    ta_f = float((torch.exp(log_ta).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    return HybridNativeTemperatureScale(T_ang=ta_f, T_gauss_x=tg_f, T_gauss_y=tg_f)


def trunc_ex_tempered_joint_log_prob(
    y_g: torch.Tensor,
    psi: torch.Tensor,
    ex: torch.Tensor,
    mix_logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    mu_ex: torch.Tensor,
    sigma_ex: torch.Tensor,
    *,
    T_gauss_x: torch.Tensor,
    T_gauss_y: torch.Tensor,
    T_ang: torch.Tensor,
) -> torch.Tensor:
    """
    Joint log p for native_trunc_ex_vm with post-hoc temperatures on the **Gaussian + ψ** block
    only (same construction as ``fit_hybrid_native_temperature``): inflate (x,e_y*) covariance
    via row-scaled Cholesky and deflate von Mises κ for ψ. The truncated-normal factor for e_x
    uses the **learned** μ_ex, σ_ex unchanged (mirrors incumbent z-head, which only temp-scaled
    Gaussian + angular factors, not an extra head).
    """
    from .models_z import mdn_hybrid_gauss_vm_psi_only_log_prob, trunc_normal_01_log_prob

    Lcal = _hybrid_scaled_L(L_g, T_gauss_x, T_gauss_y)
    kpsc = kappa_psi / T_ang.clamp(min=1e-8)
    lp_g = mdn_hybrid_gauss_vm_psi_only_log_prob(y_g, psi, mix_logits, mu_g, Lcal, mu_psi, kpsc)
    lp_x = trunc_normal_01_log_prob(ex, mu_ex, sigma_ex)
    return lp_g + lp_x


def fit_hybrid_trunc_ex_temperature(
    mix_logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    mu_ex: torch.Tensor,
    sigma_ex: torch.Tensor,
    y_g: torch.Tensor,
    psi: torch.Tensor,
    ex: torch.Tensor,
    w: torch.Tensor,
    *,
    max_iter: int = 400,
    lr: float = 0.05,
    anisotropic_gaussian: bool = False,
) -> HybridNativeTemperatureScale:
    """
    Post-hoc calibration for ``ConditionalHybridGaussVonMisesTruncExZ`` on the calibration split:
    same isotropic / axis-wise Gaussian inflation + single angular T_ang on ψ as
    ``fit_hybrid_native_temperature``; joint NLL includes the fixed e_x truncated-normal term.
    """
    lg = mix_logits.detach()
    mg = mu_g.detach()
    l0 = L_g.detach()
    mp = mu_psi.detach()
    kp0 = kappa_psi.detach()
    mex = mu_ex.detach()
    sex = sigma_ex.detach()
    yd = y_g.detach()
    psid = psi.detach()
    exd = ex.detach()
    wd = w.detach()
    dev = y_g.device
    dt = y_g.dtype

    if anisotropic_gaussian:
        log_tx = torch.zeros(1, device=dev, dtype=dt, requires_grad=True)
        log_ty = torch.zeros(1, device=dev, dtype=dt, requires_grad=True)
        log_ta = torch.zeros(1, device=dev, dtype=dt, requires_grad=True)
        opt = torch.optim.Adam([log_tx, log_ty, log_ta], lr=lr)
        for _ in range(max_iter):
            opt.zero_grad()
            tx = torch.exp(log_tx).clamp(min=1e-8) + 1e-8
            ty = torch.exp(log_ty).clamp(min=1e-8) + 1e-8
            ta = torch.exp(log_ta).clamp(min=1e-8) + 1e-8
            lp = trunc_ex_tempered_joint_log_prob(
                yd,
                psid,
                exd,
                lg,
                mg,
                l0,
                mp,
                kp0,
                mex,
                sex,
                T_gauss_x=tx,
                T_gauss_y=ty,
                T_ang=ta,
            )
            nll = -lp
            loss = (nll * wd).sum() / wd.sum().clamp_min(1e-12)
            loss.backward()
            opt.step()
        tx_f = float((torch.exp(log_tx).clamp(min=1e-8) + 1e-8).detach().cpu().item())
        ty_f = float((torch.exp(log_ty).clamp(min=1e-8) + 1e-8).detach().cpu().item())
        ta_f = float((torch.exp(log_ta).clamp(min=1e-8) + 1e-8).detach().cpu().item())
        return HybridNativeTemperatureScale(T_ang=ta_f, T_gauss_x=tx_f, T_gauss_y=ty_f)

    log_tg = torch.zeros(1, device=dev, dtype=dt, requires_grad=True)
    log_ta = torch.zeros(1, device=dev, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([log_tg, log_ta], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        tg = torch.exp(log_tg).clamp(min=1e-8) + 1e-8
        ta = torch.exp(log_ta).clamp(min=1e-8) + 1e-8
        tx = ty = tg
        lp = trunc_ex_tempered_joint_log_prob(
            yd,
            psid,
            exd,
            lg,
            mg,
            l0,
            mp,
            kp0,
            mex,
            sex,
            T_gauss_x=tx,
            T_gauss_y=ty,
            T_ang=ta,
        )
        nll = -lp
        loss = (nll * wd).sum() / wd.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()

    tg_f = float((torch.exp(log_tg).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    ta_f = float((torch.exp(log_ta).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    return HybridNativeTemperatureScale(T_ang=ta_f, T_gauss_x=tg_f, T_gauss_y=tg_f)


@dataclass
class BranchAutoregTemperatureScale:
    T_x: float
    T_y: float
    T_psi: float
    T_ex: float

    @property
    def T_gauss(self) -> float:
        return float((self.T_x * self.T_y) ** 0.5)


def fit_branch_autoreg_bounded_temperature(
    model: Any,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    r_xy: torch.Tensor,
    psi: torch.Tensor,
    r_ex: torch.Tensor,
    w: torch.Tensor,
    *,
    max_iter: int = 600,
    lr: float = 0.05,
    anisotropic_xy: bool = True,
    calibrate_ex: bool = True,
) -> BranchAutoregTemperatureScale:
    """Fit post-hoc temperatures on temp_cal only (model weights frozen)."""
    with torch.no_grad():
        h = model.trunk(x_num, cat)
        logits, mu_xy, L_xy = model.mixture_params(h)
    lg = logits.detach()
    mg = mu_xy.detach()
    l0 = L_xy.detach()
    h = h.detach()
    rd = r_xy.detach()
    psid = psi.detach()
    exd = r_ex.detach()
    wd = w.detach()

    if anisotropic_xy and calibrate_ex:
        log_tx = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
        log_ty = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
        log_tp = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
        log_te = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
        opt = torch.optim.Adam([log_tx, log_ty, log_tp, log_te], lr=lr)
        for _ in range(max_iter):
            opt.zero_grad()
            tx = torch.exp(log_tx).clamp(min=1e-8) + 1e-8
            ty = torch.exp(log_ty).clamp(min=1e-8) + 1e-8
            tp = torch.exp(log_tp).clamp(min=1e-8) + 1e-8
            te = torch.exp(log_te).clamp(min=1e-8) + 1e-8
            lp = model.log_prob_model_space(rd, psid, exd, lg, mg, l0, h, T_x=tx, T_y=ty, T_psi=tp, T_ex=te)
            nll = -lp
            loss = (nll * wd).sum() / wd.sum().clamp_min(1e-12)
            loss.backward()
            opt.step()
        return BranchAutoregTemperatureScale(
            T_x=float(tx.detach().cpu().item()),
            T_y=float(ty.detach().cpu().item()),
            T_psi=float(tp.detach().cpu().item()),
            T_ex=float(te.detach().cpu().item()),
        )

    log_tg = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
    log_tp = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
    params: list[torch.Tensor] = [log_tg, log_tp]
    log_te = torch.zeros(1, device=r_xy.device, dtype=r_xy.dtype, requires_grad=True)
    if calibrate_ex:
        params.append(log_te)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        tg = torch.exp(log_tg).clamp(min=1e-8) + 1e-8
        tp = torch.exp(log_tp).clamp(min=1e-8) + 1e-8
        te = (torch.exp(log_te).clamp(min=1e-8) + 1e-8) if calibrate_ex else r_xy.new_tensor(1.0)
        lp = model.log_prob_model_space(
            rd, psid, exd, lg, mg, l0, h, T_x=tg, T_y=tg, T_psi=tp, T_ex=te
        )
        nll = -lp
        loss = (nll * wd).sum() / wd.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()
    tg_f = float((torch.exp(log_tg).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    tp_f = float((torch.exp(log_tp).clamp(min=1e-8) + 1e-8).detach().cpu().item())
    te_f = float((torch.exp(log_te).clamp(min=1e-8) + 1e-8).detach().cpu().item()) if calibrate_ex else 1.0
    return BranchAutoregTemperatureScale(T_x=tg_f, T_y=tg_f, T_psi=tp_f, T_ex=te_f)
