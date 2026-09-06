"""Forward pipeline sampling: frozen stage-u and stage-z -> raw u, z tensors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import VonMises

from .models_z import (
    BranchAutoregBoundedHybridMDNZ,
    ConditionalHybridGaussVonMisesMixtureZ,
    ConditionalHybridGaussVonMisesTruncExZ,
    sample_branch_autoreg_bounded_mdn,
    sample_mdn_gauss_psi_only,
    sample_mdn_hybrid_gauss_vm,
)
from .target_transforms_z import BoundedLogitZscoreState, model_to_raw_coords
from .models_u import SharedMixtureGaussianVonMisesUNet
from .spray_angle_bounds import SPRAY_ANGLE_DEG_LIMITS


def _sample_truncated_normal_icdf(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: float,
    high: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``N(mu, sigma)`` truncated to ``[low, high]`` via inverse CDF.

    ``torch.distributions.TruncatedNormal`` is not available on older PyTorch;
    this matches the same distribution for sampling.
    """
    sigma = sigma.clamp_min(1e-8)
    dist = torch.distributions.Normal(mu, sigma)
    lo = torch.full_like(mu, float(low))
    hi = torch.full_like(mu, float(high))
    cdf_lo = dist.cdf(lo)
    cdf_hi = dist.cdf(hi)
    span = (cdf_hi - cdf_lo).clamp_min(1e-12)
    if generator is not None:
        u = torch.rand(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    else:
        u = torch.rand_like(mu)
    p = cdf_lo + span * u.clamp(1e-12, 1.0 - 1e-12)
    return dist.icdf(p)


def _sample_u_va_d_joint_rejection(
    logits: torch.Tensor,
    mu_va: torch.Tensor,
    L_va: torch.Tensor,
    loc_d: torch.Tensor,
    kappa_d: torch.Tensor,
    *,
    n_samples: int,
    T_va: float,
    T_ang: float,
    kappa_max: float,
    d_tilde_deg_limits: tuple[float, float],
    max_resamples: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Joint (v_ss, a) Gaussian + von Mises d_tilde with rejection on d_tilde support.

    Returns ``y_std`` (S, 2) and ``d_deg`` (S,) with the same mixture index per row.
    """
    lo, hi = float(d_tilde_deg_limits[0]), float(d_tilde_deg_limits[1])
    pi = F.softmax(logits, dim=-1)
    scale = float(math.sqrt(T_va))
    y_parts: list[torch.Tensor] = []
    d_parts: list[torch.Tensor] = []
    need = int(n_samples)
    for _ in range(max(1, int(max_resamples))):
        if need <= 0:
            break
        batch = max(need, min(need * 4, 4096))
        comp = torch.multinomial(
            pi.expand(batch, -1),
            1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        mu_sel = mu_va[0, comp]
        L_sel = L_va[0, comp] * scale
        eps = torch.randn(batch, 2, device=device, dtype=dtype, generator=generator)
        y_std = mu_sel + torch.einsum("sij,sj->si", L_sel, eps)
        loc_sel = loc_d[0, comp]
        kap_sel = (kappa_d[0, comp] / float(T_ang)).clamp(min=1e-4, max=float(kappa_max))
        dist_v = VonMises(loc_sel, kap_sel, validate_args=False)
        d_rad = dist_v.sample((1,)).squeeze(0)
        d_deg = torch.rad2deg(d_rad)
        ok = (d_deg >= lo) & (d_deg <= hi)
        if ok.any():
            y_parts.append(y_std[ok])
            d_parts.append(d_deg[ok])
            need -= int(ok.sum().item())
    if not y_parts:
        raise RuntimeError(
            f"d_tilde rejection failed: no draws in [{lo}, {hi}] after {max_resamples} batches"
        )
    y_out = torch.cat(y_parts, dim=0)[:n_samples]
    d_out = torch.cat(d_parts, dim=0)[:n_samples]
    if y_out.shape[0] < n_samples:
        raise RuntimeError(
            f"d_tilde rejection failed: got {y_out.shape[0]} / {n_samples} in [{lo}, {hi}]"
        )
    return y_out, d_out


@torch.no_grad()
def sample_u_shared_gaussian_vm(
    model: SharedMixtureGaussianVonMisesUNet,
    x_u: torch.Tensor,
    cat_u: dict[str, torch.Tensor],
    *,
    n_samples: int,
    T_va: float,
    T_ang: float,
    kappa_max: float,
    target_means: torch.Tensor,
    target_stds: torch.Tensor,
    generator: torch.Generator | None = None,
    d_tilde_deg_limits: tuple[float, float] | None = None,
    max_d_tilde_resamples: int = 500,
    va_transform: Any | None = None,
    player_ids: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """
    One context row (B=1). Returns raw-space u columns (v_ss_tilde, a_tilde, d_tilde), shape (S,).

    When ``d_tilde_deg_limits`` is set (default ±45°), resample von Mises draws until all
    ``n_samples`` values lie in that interval.
    """
    logits, mu_va, L_va, loc_d, kappa_d = model(x_u, cat_u)
    if d_tilde_deg_limits is None:
        pi = F.softmax(logits, dim=-1)
        comp = torch.multinomial(
            pi.expand(n_samples, -1),
            1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        mu_sel = mu_va[0, comp]
        scale = float(math.sqrt(T_va))
        L_sel = L_va[0, comp] * scale
        eps = torch.randn(n_samples, 2, device=x_u.device, dtype=x_u.dtype, generator=generator)
        y_std = mu_sel + torch.einsum("sij,sj->si", L_sel, eps)
        loc_sel = loc_d[0, comp]
        kap_sel = (kappa_d[0, comp] / float(T_ang)).clamp(min=1e-4, max=float(kappa_max))
        dist_v = VonMises(loc_sel, kap_sel, validate_args=False)
        d_rad = dist_v.sample((1,)).squeeze(0)
        d_deg = torch.rad2deg(d_rad)
    else:
        y_std, d_deg = _sample_u_va_d_joint_rejection(
            logits,
            mu_va,
            L_va,
            loc_d,
            kappa_d,
            n_samples=n_samples,
            T_va=T_va,
            T_ang=T_ang,
            kappa_max=kappa_max,
            d_tilde_deg_limits=d_tilde_deg_limits,
            max_resamples=max_d_tilde_resamples,
            device=x_u.device,
            dtype=x_u.dtype,
            generator=generator,
        )
    tm = target_means.view(1, 3).to(device=x_u.device, dtype=x_u.dtype)
    ts = target_stds.view(1, 3).to(device=x_u.device, dtype=x_u.dtype)
    if va_transform is not None:
        if player_ids is None:
            raise ValueError("player_ids are required when va_transform is provided")
        y_np = y_std[:, :2].detach().cpu().numpy()
        raw_va = va_transform.inverse(y_np, np.asarray(player_ids, dtype=np.int64))
        raw0 = torch.as_tensor(raw_va[:, 0], device=x_u.device, dtype=x_u.dtype)
        raw1 = torch.as_tensor(raw_va[:, 1], device=x_u.device, dtype=x_u.dtype)
    else:
        raw0 = y_std[:, 0] * ts[0, 0] + tm[0, 0]
        raw1 = y_std[:, 1] * ts[0, 1] + tm[0, 1]
    raw2 = d_deg
    return {"v_ss_tilde": raw0, "a_tilde": raw1, "d_tilde": raw2}


@torch.no_grad()
def sample_z_hybrid_native_batched(
    model: ConditionalHybridGaussVonMisesMixtureZ,
    x_z: torch.Tensor,
    cat_z: dict[str, torch.Tensor],
    *,
    n_samples_per_row: int,
    T_gauss: float,
    T_gauss_x: float | None,
    T_gauss_y: float | None,
    T_ang: float,
    gauss_means: torch.Tensor,
    gauss_stds: torch.Tensor,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """
    Batch B rows (different contexts). Returns raw z per row: x, psi_deg, e_y_star, theta_deg.
    Each row gets ``n_samples_per_row`` i.i.d. samples → shapes (B, S).
    """
    from .calibration_z import _hybrid_scaled_L

    logits, mu_g, L_g, mu_psi, kappa_psi, mu_theta, kappa_theta = model(x_z, cat_z)
    if T_gauss_x is not None and T_gauss_y is not None:
        tx = L_g.new_tensor(float(T_gauss_x))
        ty = L_g.new_tensor(float(T_gauss_y))
        Lcal = _hybrid_scaled_L(L_g, tx, ty)
    else:
        Lcal = L_g * float(np.sqrt(T_gauss))
    kpsc = kappa_psi / float(T_ang)
    kthc = kappa_theta / float(T_ang)
    y_g, psi_s, th_s = sample_mdn_hybrid_gauss_vm(
        logits,
        mu_g,
        Lcal,
        mu_psi,
        kpsc,
        mu_theta,
        kthc,
        n_samples_per_row,
        generator=generator,
    )
    gm = gauss_means.view(1, 2).to(device=x_z.device, dtype=x_z.dtype)
    gs = gauss_stds.view(1, 2).to(device=x_z.device, dtype=x_z.dtype)
    B, S, _ = y_g.shape
    x_raw = y_g[..., 0] * gs[0, 0] + gm[0, 0]
    ey_raw = y_g[..., 1] * gs[0, 1] + gm[0, 1]
    psi_deg = torch.rad2deg(psi_s)
    theta_deg = torch.rad2deg(th_s)
    return {
        "x": x_raw,
        "e_y_star": ey_raw,
        "psi_deg": psi_deg,
        "theta_deg": theta_deg,
    }


@torch.no_grad()
def sample_z_hybrid_trunc_ex_batched(
    model: ConditionalHybridGaussVonMisesTruncExZ,
    x_z: torch.Tensor,
    catz: dict[str, torch.Tensor],
    *,
    n_samples_per_row: int,
    T_gauss: float,
    T_gauss_x: float | None,
    T_gauss_y: float | None,
    T_ang: float,
    gauss_means: torch.Tensor,
    gauss_stds: torch.Tensor,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample (x, e_y*, psi, e_x) for ``native_trunc_ex_vm`` checkpoints (theta-free)."""
    from .calibration_z import _hybrid_scaled_L

    logits, mu_g, L_g, mu_psi, kappa_psi, h = model(x_z, catz)
    if T_gauss_x is not None and T_gauss_y is not None:
        tx = L_g.new_tensor(float(T_gauss_x))
        ty = L_g.new_tensor(float(T_gauss_y))
        Lcal = _hybrid_scaled_L(L_g, tx, ty)
    else:
        Lcal = L_g * float(np.sqrt(T_gauss))
    kpsc = kappa_psi / float(T_ang)
    y_g, psi_s = sample_mdn_gauss_psi_only(
        logits,
        mu_g,
        Lcal,
        mu_psi,
        kpsc,
        n_samples_per_row,
        generator=generator,
    )
    mex, sig = model.ex_params_batched(h, y_g, psi_s)
    sig_b = sig.clamp_min(1e-4)
    ex_flat = _sample_truncated_normal_icdf(mex, sig_b, 0.0, 0.6, generator=generator)

    gm = gauss_means.view(1, 2).to(device=x_z.device, dtype=x_z.dtype)
    gs = gauss_stds.view(1, 2).to(device=x_z.device, dtype=x_z.dtype)
    B, S, _ = y_g.shape
    ex_s = ex_flat.view(B, S)
    x_raw = y_g[..., 0] * gs[0, 0] + gm[0, 0]
    ey_raw = y_g[..., 1] * gs[0, 1] + gm[0, 1]
    psi_deg = torch.rad2deg(psi_s)
    return {
        "x": x_raw,
        "e_y_star": ey_raw,
        "psi_deg": psi_deg,
        "e_x": ex_s,
    }


@torch.no_grad()
def sample_z_branch_autoreg_bounded_mdn_batched(
    model: BranchAutoregBoundedHybridMDNZ,
    x_z: torch.Tensor,
    catz: dict[str, torch.Tensor],
    *,
    n_samples_per_row: int,
    transform_state: BoundedLogitZscoreState | dict[str, Any],
    x_lower: torch.Tensor,
    x_upper: torch.Tensor,
    T_x: float = 1.0,
    T_y: float = 1.0,
    T_psi: float = 1.0,
    T_ex: float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample (x, psi_deg, e_y_star, e_x) for branch_autoreg_bounded_hybrid_mdn_z checkpoints."""
    if isinstance(transform_state, dict):
        st = BoundedLogitZscoreState(
            float(transform_state["mean_r_x"]),
            float(transform_state["std_r_x"]),
            float(transform_state["mean_r_y"]),
            float(transform_state["std_r_y"]),
            float(transform_state["mean_r_ex"]),
            float(transform_state["std_r_ex"]),
            float(transform_state.get("eps", 1e-5)),
            str(transform_state.get("x_support_mode", "physics")),
            bool(transform_state.get("x_used_empirical_fallback", False)),
            float(transform_state.get("e_y_star_bounds", [0, 1])[0]),
            float(transform_state.get("e_y_star_bounds", [0, 1])[1]),
            float(transform_state.get("e_x_bounds", [0, 0.6])[0]),
            float(transform_state.get("e_x_bounds", [0, 0.6])[1]),
            float(transform_state.get("e_x_scale", 0.6)),
        )
    else:
        st = transform_state

    r_xy, psi_s, r_ex = sample_branch_autoreg_bounded_mdn(
        model,
        x_z,
        catz,
        n_samples=n_samples_per_row,
        T_x=T_x,
        T_y=T_y,
        T_psi=T_psi,
        T_ex=T_ex,
        generator=generator,
    )
    B, S, _ = r_xy.shape
    x_out = torch.empty(B, S, device=x_z.device, dtype=x_z.dtype)
    ey_out = torch.empty(B, S, device=x_z.device, dtype=x_z.dtype)
    ex_out = torch.empty(B, S, device=x_z.device, dtype=x_z.dtype)
    for b in range(B):
        for s in range(S):
            xr, eyr, exr = model_to_raw_coords(
                float(r_xy[b, s, 0].cpu()),
                float(r_xy[b, s, 1].cpu()),
                float(r_ex[b, s].cpu()),
                x_lower=x_lower[b : b + 1].cpu().numpy(),
                x_upper=x_upper[b : b + 1].cpu().numpy(),
                state=st,
            )
            x_out[b, s] = float(xr[0])
            ey_out[b, s] = float(eyr[0])
            ex_out[b, s] = float(exr[0])
    return {
        "x": x_out,
        "e_y_star": ey_out,
        "psi_deg": torch.rad2deg(psi_s),
        "e_x": ex_out,
    }


def patch_z_x_num_with_u(
    base_x_z: np.ndarray,
    num_order: list[str],
    u_cols: tuple[str, str, str],
    u_raw: np.ndarray,
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    """base_x_z: (F,) standardized z features; u_raw: (S,3) v_ss, a_tilde, d_tilde."""
    idx = [num_order.index(c) for c in u_cols]
    out = np.tile(base_x_z.astype(np.float64), (u_raw.shape[0], 1))
    for j, col in enumerate(u_cols):
        mu = float(stats[col]["mean"])
        sig = float(max(stats[col]["std"], 1e-8))
        out[:, idx[j]] = (u_raw[:, j].astype(np.float64) - mu) / sig
    return out.astype(np.float32)
