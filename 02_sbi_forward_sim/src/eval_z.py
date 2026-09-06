"""Metrics for stage-z full-covariance Gaussian mixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from .models_z import mixture_marginal_std_from_chol, mdn_log_prob_chol

Z_NAMES = ("x", "psi_deg", "e_y_star", "theta_deg")


def weighted_mdn_nll_chol_torch(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    T_mix: float = 1.0,
) -> tuple[float, float]:
    lcal = L * float(np.sqrt(T_mix))
    lp = mdn_log_prob_chol(y, mix_logits, mu, lcal)
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    return float((nll * w).sum().item() / wsum), float(nll.mean().item())


def _marginal_mixture_cdf(
    y: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    marg_std: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """y (B,), pi (B,K), mu (B,K,D), marg_std (B,K,D) = sqrt(Sigma_jj)."""
    yk = y.unsqueeze(-1)
    m = mu[:, :, dim]
    s = marg_std[:, :, dim].clamp_min(1e-8)
    z = (yk - m) / s
    Phi = torch.special.ndtr(z)
    return (pi * Phi).sum(dim=-1)


@torch.no_grad()
def evaluate_z_split_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_std: torch.Tensor,
    y_raw: torch.Tensor,
    w: torch.Tensor,
    target_means: torch.Tensor,
    target_stds: torch.Tensor,
    *,
    T_mix: float = 1.0,
    batch_size: int = 4096,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    n = x_num.shape[0]
    tm = target_means.view(1, 4)
    ts = target_stds.view(1, 4)
    D = 4

    sum_w_nll = 0.0
    sum_u_nll = 0.0
    sum_w = 0.0
    sum_abs_err = torch.zeros(4, device=device)
    sum_sq_err = torch.zeros(4, device=device)
    cnt_cov50 = torch.zeros(4, device=device)
    cnt_cov80 = torch.zeros(4, device=device)
    cnt_cov90 = torch.zeros(4, device=device)
    sum_pred_var_raw = torch.zeros(4, device=device)
    pits_parts: list[np.ndarray] = []

    z50 = float(stats.norm.ppf(0.75))
    z80 = float(stats.norm.ppf(0.9))
    z90 = float(stats.norm.ppf(0.95))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        xb = x_num[sl]
        cb = {k: v[sl] for k, v in cat.items()}
        yb_std = y_std[sl]
        yb_raw = y_raw[sl]
        wb = w[sl]
        logits, mu, L = model(xb, cb)
        lcal = L * float(np.sqrt(T_mix))
        lp = mdn_log_prob_chol(yb_std, logits, mu, lcal)
        nll = -lp
        sum_w_nll += float((nll * wb).sum().item())
        sum_u_nll += float(nll.sum().item())
        sum_w += float(wb.sum().item())

        pi = torch.nn.functional.softmax(logits, dim=-1)
        mu_pred = (pi.unsqueeze(-1) * mu).sum(dim=1)
        marg_std = mixture_marginal_std_from_chol(lcal)
        ex2 = (pi.unsqueeze(-1) * (mu**2 + marg_std**2)).sum(dim=1)
        var_m = (ex2 - mu_pred**2).clamp_min(1e-12)
        std_m = torch.sqrt(var_m)

        mu_raw = mu_pred * ts + tm
        err = yb_raw - mu_raw
        sum_abs_err += torch.sum(torch.abs(err), dim=0)
        sum_sq_err += torch.sum(err**2, dim=0)

        for k in range(D):
            low50 = (mu_pred[:, k] - z50 * std_m[:, k]) * ts[0, k] + tm[0, k]
            high50 = (mu_pred[:, k] + z50 * std_m[:, k]) * ts[0, k] + tm[0, k]
            low80 = (mu_pred[:, k] - z80 * std_m[:, k]) * ts[0, k] + tm[0, k]
            high80 = (mu_pred[:, k] + z80 * std_m[:, k]) * ts[0, k] + tm[0, k]
            low90 = (mu_pred[:, k] - z90 * std_m[:, k]) * ts[0, k] + tm[0, k]
            high90 = (mu_pred[:, k] + z90 * std_m[:, k]) * ts[0, k] + tm[0, k]
            yk = yb_raw[:, k]
            cnt_cov50[k] += float(((yk >= low50) & (yk <= high50)).sum().item())
            cnt_cov80[k] += float(((yk >= low80) & (yk <= high80)).sum().item())
            cnt_cov90[k] += float(((yk >= low90) & (yk <= high90)).sum().item())

        sum_pred_var_raw += torch.sum(var_m * (ts**2), dim=0)

        pit_cols = []
        for k in range(D):
            pit_cols.append(
                _marginal_mixture_cdf(yb_std[:, k], pi, mu, marg_std, k).clamp(0.0, 1.0).unsqueeze(1)
            )
        pits_parts.append(torch.cat(pit_cols, dim=1).cpu().numpy())

    inv = 1.0 / float(max(n, 1))
    pits = np.concatenate(pits_parts, axis=0)
    hist_counts = [np.histogram(pits[:, k], bins=20, range=(0, 1))[0].tolist() for k in range(4)]

    return {
        "n_rows": int(n),
        "weighted_nll": sum_w_nll / max(sum_w, 1e-12),
        "unweighted_nll": sum_u_nll / max(n, 1),
        "mae_raw": (sum_abs_err * inv).cpu().numpy().tolist(),
        "rmse_raw": torch.sqrt(sum_sq_err * inv).cpu().numpy().tolist(),
        "target_names": list(Z_NAMES),
        "coverage_50_marginal": (cnt_cov50 * inv).cpu().numpy().tolist(),
        "coverage_80_marginal": (cnt_cov80 * inv).cpu().numpy().tolist(),
        "coverage_90_marginal": (cnt_cov90 * inv).cpu().numpy().tolist(),
        "avg_pred_std_marginal_raw": torch.sqrt(sum_pred_var_raw * inv).cpu().numpy().tolist(),
        "empirical_std_raw": np.std(y_raw.detach().cpu().numpy(), axis=0).tolist(),
        "pit_hist_counts": hist_counts,
        "eval_note": (
            "Full-covariance mixture; marginal PIT uses componentwise Gaussian CDFs with "
            "sqrt(diag(Sigma_k)) from Cholesky. Intervals use mixture marginal mean/variance."
        ),
    }


def save_metrics_json_csv(out_dir: Path, name: str, pack: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    import csv

    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in pack.items()}
    with (out_dir / f"{name}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(flat.keys())
        w.writerow(flat.values())


@torch.no_grad()
def collect_pits_z_batched(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_std: torch.Tensor,
    *,
    T_mix: float = 1.0,
    batch_size: int = 8192,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = x_num.shape[0]
    parts: list[np.ndarray] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        logits, mu, L = model(x_num[s:e], {k: v[s:e] for k, v in cat.items()})
        lcal = L * float(np.sqrt(T_mix))
        pi = torch.nn.functional.softmax(logits, dim=-1)
        marg_std = mixture_marginal_std_from_chol(lcal)
        pit_cols = []
        for k in range(4):
            pit_cols.append(
                _marginal_mixture_cdf(y_std[s:e, k], pi, mu, marg_std, k).clamp(0.0, 1.0).unsqueeze(1)
            )
        parts.append(torch.cat(pit_cols, dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def plot_pit_hist_z(pits: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for k, ax in enumerate(axes):
        ax.hist(pits[:, k], bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax.set_title(Z_NAMES[k])
        ax.set_xlabel("PIT")
    fig.suptitle("Stage-z marginal PIT (full-covariance GMM)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
