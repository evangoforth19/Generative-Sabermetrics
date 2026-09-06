"""Training loop for branch_autoreg_bounded_hybrid_mdn_z."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration_z import BranchAutoregTemperatureScale, fit_branch_autoreg_bounded_temperature
from .data_u import index_by_event
from .data_z_branch_autoreg import ZArraysBranchAutoreg
from .models_z import BranchAutoregBoundedHybridMDNZ


def _to_tensors_branch(arr: ZArraysBranchAutoreg, device: torch.device):
    x = torch.from_numpy(arr.x_num).to(device)
    cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    r_xy = torch.from_numpy(arr.r_xy_std).to(device)
    psi = torch.from_numpy(arr.psi_rad).to(device)
    r_ex = torch.from_numpy(arr.r_ex_std).to(device)
    j_log = torch.from_numpy(arr.j_log_raw).to(device)
    w = torch.from_numpy(arr.w).to(device)
    return x, cat, r_xy, psi, r_ex, j_log, w


def _batch_nll_and_loss(
    model: BranchAutoregBoundedHybridMDNZ,
    x: torch.Tensor,
    cat: dict[str, torch.Tensor],
    r_xy: torch.Tensor,
    psi: torch.Tensor,
    r_ex: torch.Tensor,
    w: torch.Tensor,
    cfg: dict[str, Any],
    *,
    temp: BranchAutoregTemperatureScale | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    h = model.trunk(x, cat)
    logits, mu_xy, L_xy = model.mixture_params(h)
    kw = {}
    if temp is not None:
        kw = dict(T_x=temp.T_x, T_y=temp.T_y, T_psi=temp.T_psi, T_ex=temp.T_ex)
    lp, fac = model.log_prob_components(r_xy, psi, r_ex, logits, mu_xy, L_xy, h, **kw)
    nll = -lp
    loss = (nll * w).sum() / w.sum().clamp_min(1e-12)
    reg = model.regularization_terms(logits, L_xy)
    rcfg = cfg.get("regularization", {})
    loss = loss + float(rcfg.get("lambda_branch_embedding_l2", 0.0)) * reg["branch_emb_l2"]
    loss = loss + float(rcfg.get("lambda_cholesky_offdiag_l2", 0.0)) * reg["offdiag_l2"]
    loss = loss + float(rcfg.get("lambda_log_scale_l2", 0.0)) * reg["log_diag_l2"]
    lam_ent = float(rcfg.get("lambda_mixture_entropy", 0.0))
    if lam_ent != 0.0:
        loss = loss - lam_ent * reg["mixture_entropy"]
    diag = {
        "nll_model": float(nll.detach().mean().item()),
        "fac_xy": float(fac["xy"].detach().mean().item()),
        "fac_psi": float(fac["psi"].detach().mean().item()),
        "fac_ex": float(fac["ex"].detach().mean().item()),
    }
    return loss, diag


@torch.no_grad()
def weighted_eval_nll(
    model: BranchAutoregBoundedHybridMDNZ,
    arr: ZArraysBranchAutoreg,
    device: torch.device,
    batch_size: int = 4096,
    *,
    temp: BranchAutoregTemperatureScale | None = None,
) -> float:
    model.eval()
    x, cat, r_xy, psi, r_ex, _, w = _to_tensors_branch(arr, device)
    n = x.shape[0]
    wsum = 0.0
    wnll = 0.0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sl = slice(s, e)
        bx = x[sl]
        bc = {k: v[sl] for k, v in cat.items()}
        h = model.trunk(bx, bc)
        logits, mu_xy, L_xy = model.mixture_params(h)
        kw = {}
        if temp is not None:
            kw = dict(T_x=temp.T_x, T_y=temp.T_y, T_psi=temp.T_psi, T_ex=temp.T_ex)
        lp, _ = model.log_prob_components(
            r_xy[sl], psi[sl], r_ex[sl], logits, mu_xy, L_xy, h, **kw
        )
        wx = w[sl]
        wsum += float(wx.sum().item())
        wnll += float(((-lp) * wx).sum().item())
    return wnll / max(wsum, 1e-12)


def train_stage_z_branch_autoreg_bounded_mdn(
    cfg: dict[str, Any],
    train_a: ZArraysBranchAutoreg,
    val_a: ZArraysBranchAutoreg,
    temp_a: ZArraysBranchAutoreg,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    cov = cfg.get("covariance", {})
    ex_cfg = cfg.get("ex", {})
    torch.manual_seed(int(tcfg["seed"]))
    np.random.seed(int(tcfg["seed"]))

    x_tr, cat_tr, rxy_tr, psi_tr, rex_tr, _, w_tr = _to_tensors_branch(train_a, device)
    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(int(tcfg["seed"]))

    num_f = train_a.x_num.shape[1]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = BranchAutoregBoundedHybridMDNZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=int(mcfg.get("K", mcfg.get("n_components", 5))),
        hidden_width=int(mcfg.get("hidden_dim", mcfg.get("hidden_width", 64))),
        n_hidden=int(mcfg.get("depth", mcfg.get("hidden_layers", 2))),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(mcfg.get("dropout", 0.0)),
        branch_embedding_dim=int(mcfg.get("branch_embedding_dim", 8)),
        chol_eps=float(cov.get("chol_eps", 1e-4)),
        diag_floor=float(cov.get("diag_floor", 0.05)),
        diag_ceiling=float(cov.get("diag_ceiling", 5.0)),
        use_diag_ceiling=bool(cov.get("use_diag_ceiling", True)),
        offdiag_tanh_scale=float(cov.get("offdiag_tanh_scale", 2.0)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=float(mcfg.get("kappa_max", 120.0)),
        sigma_floor=float(ex_cfg.get("sigma_floor", 0.05)),
        sigma_ceiling=float(ex_cfg.get("sigma_ceiling", 5.0)),
        use_sigma_ceiling=bool(ex_cfg.get("use_sigma_ceiling", True)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(cfg.get("regularization", {}).get("optimizer_weight_decay", tcfg.get("weight_decay", 3e-4))),
    )

    batch_events = int(tcfg.get("batch_events", 64))
    max_draws = int(tcfg.get("draws_per_event", tcfg.get("max_draws_per_event", 16)))
    max_epochs = 1 if tcfg.get("debug_one_epoch") else int(tcfg["max_epochs"])
    patience = int(tcfg["patience"])
    best_val = float("inf")
    bad = 0
    best_state = None
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(event_list)
        perm_events = list(event_list)
        epoch_loss = 0.0
        n_batches = 0
        pos = 0
        while pos < len(perm_events):
            batch_ev = perm_events[pos : pos + batch_events]
            pos += batch_events
            ix: list[np.ndarray] = []
            for e in batch_ev:
                pool = idx_by_event[e]
                if len(pool) <= max_draws:
                    ix.append(pool)
                else:
                    ix.append(rng.choice(pool, size=max_draws, replace=False))
            if not ix:
                continue
            bix = np.concatenate(ix, axis=0)
            loss, _ = _batch_nll_and_loss(
                model,
                x_tr[bix],
                {k: cat_tr[k][bix] for k in cat_tr},
                rxy_tr[bix],
                psi_tr[bix],
                rex_tr[bix],
                w_tr[bix],
                cfg,
            )
            opt.zero_grad()
            loss.backward()
            if tcfg.get("grad_clip_norm") or tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(tcfg.get("grad_clip_norm", tcfg.get("grad_clip", 5.0)))
                )
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_wnll = weighted_eval_nll(model, val_a, device)
        if val_wnll < best_val:
            best_val = val_wnll
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(
            f"epoch {epoch+1}/{max_epochs} train~{epoch_loss/max(n_batches,1):.4f} "
            f"val_select_wNLL={val_wnll:.4f} best={best_val:.4f} patience={bad}/{patience}"
        )
        if bad >= patience:
            print("Early stopping on val_select.")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0
    ccal = cfg.get("calibration", {})
    temp_scale: BranchAutoregTemperatureScale | None = None
    cal_steps = int(ccal.get("steps", ccal.get("max_iter", 600)))
    if tcfg.get("debug_one_epoch"):
        cal_steps = min(cal_steps, 20)
    if ccal.get("fit_temperatures", True):
        x_t, cat_t, rxy_t, psi_t, rex_t, _, w_t = _to_tensors_branch(temp_a, device)
        temp_scale = fit_branch_autoreg_bounded_temperature(
            model,
            x_t,
            cat_t,
            rxy_t,
            psi_t,
            rex_t,
            w_t,
            max_iter=cal_steps,
            lr=float(ccal.get("lr", 0.05)),
            anisotropic_xy=bool(ccal.get("anisotropic_xy", True)),
            calibrate_ex=bool(ccal.get("calibrate_ex", True)),
        )
    else:
        temp_scale = BranchAutoregTemperatureScale(1.0, 1.0, 1.0, 1.0)

    train_sec_total = time.time() - t0
    tm = train_a.transform_state
    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "vocabs": vocabs,
        "config": cfg,
        "numeric_feature_names": train_a.feature_names,
        "model_family": "branch_autoreg_bounded_hybrid_mdn_z",
        "target_parameterization": "branch_autoreg_bounded_hybrid_mdn_z",
        "transform_state": tm.to_dict(),
        "temperature_T_x": temp_scale.T_x,
        "temperature_T_y": temp_scale.T_y,
        "temperature_T_psi": temp_scale.T_psi,
        "temperature_T_ex": temp_scale.T_ex,
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_val_select_weighted_nll_pre_temp": best_val,
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_x": temp_scale.T_x,
                "T_y": temp_scale.T_y,
                "T_psi": temp_scale.T_psi,
                "T_ex": temp_scale.T_ex,
                "T_gauss": temp_scale.T_gauss,
                "note": "Anisotropic xy Cholesky row scaling; kappa_psi /= T_psi; sigma_ex *= sqrt(T_ex).",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_val_select_weighted_nll_pre_temp": best_val,
        "temperature": temp_scale,
    }
