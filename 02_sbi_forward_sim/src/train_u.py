"""Training loop for stage-u Gaussian model (event-balanced batches)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration_u import (
    fit_global_covariance_temperature,
    fit_hybrid_va_gmm_temperatures,
    fit_hybrid_va_vm_temperatures,
)
from .data_u import UArrays, index_by_event
from .eval_u import (
    weighted_gaussian_nll_torch,
    weighted_hybrid_joint_nll_torch,
    weighted_shared_gaussian_vm_nll_torch,
)
from .models_u import (
    BoundedHierSharedMixtureGaussianVonMisesUNet,
    BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet,
    GaussianCholeskyNet,
    HybridBivariateGaussianMixtureDNet,
    SharedMixtureGaussianVonMisesUNet,
    mixture_log_prob_1d,
)
from .target_transforms_u import (
    BoundedLogitZScoreTransform,
    PlayerSpecificBoundedLogitZScoreTransform,
)

FROZEN_BOUNDED_SUPPORT_HEAD = "shared_gaussian_vm_d_bounded_support"


def _arrays_to_tensors(arr: UArrays, device: torch.device) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x = torch.from_numpy(arr.x_num).to(device)
    cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    y = torch.from_numpy(arr.y).to(device)
    yr = torch.from_numpy(arr.y_raw).to(device)
    w = torch.from_numpy(arr.w).to(device)
    return x, cat, y, yr, w


def _arrays_to_tensors_bounded(
    arr: UArrays, device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, cat, y, yr, w = _arrays_to_tensors(arr, device)
    if arr.batter_id is None:
        raise ValueError("bounded hierarchical model requires batter_id on UArrays")
    bid = torch.from_numpy(arr.batter_id).long().to(device)
    return x, cat, y, yr, w, bid


def _ordinary_cat_dict(cat: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Categorical tensors excluding batter_name (hierarchical path)."""
    return {k: v for k, v in cat.items() if k != "batter_name"}


def build_bounded_player_support_model(
    cfg: dict[str, Any],
    num_f: int,
    vocabs: dict[str, dict[str, int]],
    device: torch.device,
) -> BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet:
    mcfg = cfg["model"]
    cvcfg = cfg.get("covariance", {})
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in cfg["categorical_features"]}
    vocab_sizes_ord = {k: len(vocabs[k]) for k in ("pitch_type", "stand", "p_throws")}
    return BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        n_batters=len(vocabs["batter_name"]),
        vocab_sizes=vocab_sizes_ord,
        embedding_dims=emb_dims,
        n_mixture=_n_mixture_from_cfg(mcfg),
        hidden_width=int(mcfg.get("hidden_width", mcfg.get("hidden_dim", 64))),
        n_hidden=int(mcfg.get("hidden_layers", mcfg.get("depth", 2))),
        activation=mcfg.get("activation", "tanh"),
        dropout=cfg["training"].get("dropout", 0.0),
        player_embedding_dim=int(mcfg.get("player_embedding_dim", 8)),
        player_effect_scale=float(mcfg.get("player_effect_scale", 1.0)),
        chol_eps=float(cvcfg.get("chol_eps", 1e-4)),
        diag_floor=float(cvcfg.get("diag_floor", 0.05)),
        diag_ceiling=float(cvcfg.get("diag_ceiling", 5.0)),
        use_diag_ceiling=bool(cvcfg.get("use_diag_ceiling", True)),
        offdiag_tanh_scale=float(cvcfg.get("offdiag_tanh_scale", 2.0)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=float(mcfg.get("kappa_max", 120.0)),
    ).to(device)


def _bounded_hier_regularization(
    model: BoundedHierSharedMixtureGaussianVonMisesUNet,
    logits: torch.Tensor,
    rcfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Explicit L2 / entropy penalties (not AdamW on embeddings)."""
    reg_terms: dict[str, float] = {}
    total = torch.zeros((), device=logits.device, dtype=logits.dtype)

    lam_pe = float(rcfg.get("lambda_player_embedding_l2", 0.0))
    if lam_pe > 0:
        pe_pen = model.player_embedding.weight.pow(2).sum()
        total = total + lam_pe * pe_pen
        reg_terms["player_embedding_l2"] = float(pe_pen.detach().item())

    lam_pa = float(rcfg.get("lambda_player_adapter_l2", 0.0))
    if lam_pa > 0:
        pa_pen = model.player_adapter.weight.pow(2).sum()
        total = total + lam_pa * pa_pen
        reg_terms["player_adapter_l2"] = float(pa_pen.detach().item())

    lam_off = float(rcfg.get("lambda_cholesky_offdiag_l2", 0.0))
    if lam_off > 0 and hasattr(model, "_last_chol_offdiag"):
        off = model._last_chol_offdiag
        off_pen = off.pow(2).mean()
        total = total + lam_off * off_pen
        reg_terms["chol_offdiag_l2_mean"] = float(off_pen.detach().item())

    lam_log = float(rcfg.get("lambda_log_scale_l2", 0.0))
    if lam_log > 0 and hasattr(model, "_last_chol_diag"):
        ld = model._last_chol_diag
        log_pen = ld.pow(2).mean()
        total = total + lam_log * log_pen
        reg_terms["log_diag_l2_mean"] = float(log_pen.detach().item())

    lam_ent = float(rcfg.get("lambda_mixture_entropy", 0.0))
    if lam_ent != 0.0:
        pi = torch.softmax(logits, dim=-1)
        ent = BoundedHierSharedMixtureGaussianVonMisesUNet.mixture_entropy(pi).mean()
        total = total - lam_ent * ent
        reg_terms["mixture_entropy_mean"] = float(ent.detach().item())

    return total, reg_terms


def _explicit_l1_l2_regularization(
    model: torch.nn.Module,
    rcfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Auditable explicit L1/L2 penalties for validation-selected regularization."""
    reg_terms: dict[str, float] = {}
    params: list[torch.Tensor] = []
    include_bias = bool(rcfg.get("include_bias", False))
    include_embeddings = bool(rcfg.get("include_embeddings", True))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not include_bias and name.endswith(".bias"):
            continue
        if not include_embeddings and name.startswith("embeddings."):
            continue
        params.append(p)

    if not params:
        total = next(model.parameters()).new_zeros(())
        return total, reg_terms

    total = params[0].new_zeros(())
    lam_l1 = float(rcfg.get("lambda_l1", rcfg.get("l1", 0.0)))
    lam_l2 = float(rcfg.get("lambda_l2", rcfg.get("l2", 0.0)))
    if lam_l1 > 0.0:
        l1_pen = sum(p.abs().sum() for p in params)
        total = total + lam_l1 * l1_pen
        reg_terms["l1_sum"] = float(l1_pen.detach().item())
        reg_terms["lambda_l1"] = lam_l1
    if lam_l2 > 0.0:
        l2_pen = sum(p.pow(2).sum() for p in params)
        total = total + lam_l2 * l2_pen
        reg_terms["l2_sum"] = float(l2_pen.detach().item())
        reg_terms["lambda_l2"] = lam_l2
    if lam_l1 > 0.0 or lam_l2 > 0.0:
        reg_terms["n_regularized_parameter_tensors"] = float(len(params))
    return total, reg_terms


def _n_mixture_from_cfg(mcfg: dict[str, Any]) -> int:
    if "n_mixture_components" in mcfg:
        return int(mcfg["n_mixture_components"])
    if "K" in mcfg:
        return int(mcfg["K"])
    raise KeyError("model config needs n_mixture_components or K")


def train_stage_u(
    cfg: dict[str, Any],
    train_a: UArrays,
    cal_a: UArrays,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    x_tr, cat_tr, y_tr, yraw_tr, w_tr = _arrays_to_tensors(train_a, device)
    x_cal, cat_cal, y_cal, yraw_cal, w_cal = _arrays_to_tensors(cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = GaussianCholeskyNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        hidden_width=mcfg["hidden_width"],
        n_hidden=mcfg["hidden_layers"],
        activation=mcfg.get("activation", "tanh"),
        dropout=tcfg.get("dropout", 0.0),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    batch_events = int(tcfg["batch_events"])
    max_draws = int(tcfg["max_draws_per_event"])

    best_cal = float("inf")
    patience = int(tcfg["early_stopping_patience"])
    bad = 0
    best_state = None
    t0 = time.time()

    for epoch in range(int(tcfg["max_epochs"])):
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
            bx = x_tr[bix]
            bc = {k: cat_tr[k][bix] for k in cat_tr}
            by = y_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            mu, tril = model(bx, bc)
            dist = torch.distributions.MultivariateNormal(
                mu, scale_tril=tril, validate_args=False
            )
            lp = dist.log_prob(by)
            nll = -lp
            loss = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches produced")

        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                mu_c, L_c = model(x_cal, cat_cal)
            wn, _ = weighted_gaussian_nll_torch(mu_c, L_c, y_cal, w_cal)
            if wn < best_cal:
                best_cal = wn
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch {epoch+1}/{tcfg['max_epochs']} train_loss~{epoch_loss/n_batches:.4f} "
                f"cal_wNLL={wn:.4f} best={best_cal:.4f} patience={bad}/{patience}"
            )
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0

    model.eval()
    with torch.no_grad():
        mu_c, L_c = model(x_cal, cat_cal)
    ccal = cfg["calibration"]
    temp = fit_global_covariance_temperature(
        mu_c.detach(),
        L_c.detach(),
        y_cal,
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
    )
    train_sec_total = time.time() - t0

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "config": cfg,
            "target_means": train_a.target_means.tolist(),
            "target_stds": train_a.target_stds.tolist(),
            "numeric_feature_names": train_a.feature_names,
            "head_family": "full_gaussian_cholesky",
            "temperature_T": temp.T,
            "train_loop_seconds": train_loop_sec,
            "train_total_seconds_including_calibration": train_sec_total,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "temperature.json").write_text(
        json.dumps({"T": temp.T, "note": "Sigma_cal = T * Sigma; scale_tril *= sqrt(T)"}, indent=2),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "covariance_temperature_T": temp.T,
    }


def train_stage_u_hybrid_gmm_d(
    cfg: dict[str, Any],
    train_a: UArrays,
    cal_a: UArrays,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Train hybrid (v_ss,a) Gaussian × d_tilde GMM; dual scalar post-hoc temperatures."""
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    n_mix = int(mcfg["n_mixture_components"])

    x_tr, cat_tr, y_tr, yraw_tr, w_tr = _arrays_to_tensors(train_a, device)
    x_cal, cat_cal, y_cal, yraw_cal, w_cal = _arrays_to_tensors(cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = HybridBivariateGaussianMixtureDNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_mixture=n_mix,
        hidden_width=mcfg["hidden_width"],
        n_hidden=mcfg["hidden_layers"],
        activation=mcfg.get("activation", "tanh"),
        dropout=tcfg.get("dropout", 0.0),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    batch_events = int(tcfg["batch_events"])
    max_draws = int(tcfg["max_draws_per_event"])

    best_cal = float("inf")
    patience = int(tcfg["early_stopping_patience"])
    bad = 0
    best_state = None
    t0 = time.time()

    for epoch in range(int(tcfg["max_epochs"])):
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
            bx = x_tr[bix]
            bc = {k: cat_tr[k][bix] for k in cat_tr}
            by = y_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            mu_va, tril_va, mix_logits, mix_mu, mix_scale = model(bx, bc)
            dist2 = torch.distributions.MultivariateNormal(
                mu_va, scale_tril=tril_va, validate_args=False
            )
            lp2 = dist2.log_prob(by[:, :2])
            lpd = mixture_log_prob_1d(by[:, 2], mix_logits, mix_mu, mix_scale)
            lp = lp2 + lpd
            nll = -lp
            loss = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches produced")

        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                m_va, L_va, logits, mu_m, sig = model(x_cal, cat_cal)
            wn, _ = weighted_hybrid_joint_nll_torch(m_va, L_va, logits, mu_m, sig, y_cal, w_cal)
            if wn < best_cal:
                best_cal = wn
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch {epoch+1}/{tcfg['max_epochs']} train_loss~{epoch_loss/n_batches:.4f} "
                f"cal_wNLL={wn:.4f} best={best_cal:.4f} patience={bad}/{patience}"
            )
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0

    model.eval()
    with torch.no_grad():
        m_va, L_va, logits, mu_m, sig = model(x_cal, cat_cal)
    ccal = cfg["calibration"]
    hybrid_temp = fit_hybrid_va_gmm_temperatures(
        m_va.detach(),
        L_va.detach(),
        logits.detach(),
        mu_m.detach(),
        sig.detach(),
        y_cal,
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
    )
    train_sec_total = time.time() - t0

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "config": cfg,
            "target_means": train_a.target_means.tolist(),
            "target_stds": train_a.target_stds.tolist(),
            "numeric_feature_names": train_a.feature_names,
            "head_family": "hybrid_va_gmm_d",
            "temperature_T_va": hybrid_temp.T_va,
            "temperature_T_mix": hybrid_temp.T_mix,
            "train_loop_seconds": train_loop_sec,
            "train_total_seconds_including_calibration": train_sec_total,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_va": hybrid_temp.T_va,
                "T_mix": hybrid_temp.T_mix,
                "note": "(v_ss,a): scale_tril *= sqrt(T_va); d mixture: σ_k *= sqrt(T_mix)",
                "head_family": "hybrid_va_gmm_d",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "covariance_temperature_T_va": hybrid_temp.T_va,
        "covariance_temperature_T_mix": hybrid_temp.T_mix,
    }


def train_stage_u_shared_gaussian_vonmises(
    cfg: dict[str, Any],
    train_a: UArrays,
    cal_a: UArrays,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Shared mixture π_k: N(v_ss,a | μ_k, Σ_k) × VonMises(d_tilde_rad | loc_k, κ_k).
    Post-hoc: T_va on Cholesky; κ_cal = κ / T_ang.
    """
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    rcfg = cfg.get("regularization", {})
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    n_mix = int(mcfg["n_mixture_components"])
    kappa_max = float(mcfg.get("kappa_max", 120.0))

    x_tr, cat_tr, y_tr, yraw_tr, w_tr = _arrays_to_tensors(train_a, device)
    x_cal, cat_cal, y_cal, yraw_cal, w_cal = _arrays_to_tensors(cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = SharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_mixture=n_mix,
        hidden_width=mcfg["hidden_width"],
        n_hidden=mcfg["hidden_layers"],
        activation=mcfg.get("activation", "tanh"),
        dropout=tcfg.get("dropout", 0.0),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=kappa_max,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    batch_events = int(tcfg["batch_events"])
    max_draws = int(tcfg["max_draws_per_event"])

    best_cal = float("inf")
    patience = int(tcfg["early_stopping_patience"])
    bad = 0
    best_state = None
    t0 = time.time()

    for epoch in range(int(tcfg["max_epochs"])):
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
            bx = x_tr[bix]
            bc = {k: cat_tr[k][bix] for k in cat_tr}
            by = y_tr[bix]
            byraw = yraw_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            logits, mu_va, L_va, loc_d, kappa_d = model(bx, bc)
            lp = model.joint_log_prob(by, byraw[:, 2], logits, mu_va, L_va, loc_d, kappa_d)
            nll = -lp
            w_nll = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            reg, reg_terms = _explicit_l1_l2_regularization(model, rcfg)
            loss = w_nll + reg
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(w_nll.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches produced")

        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                logits_c, m_va, L_va, ld, kd = model(x_cal, cat_cal)
            wn, _ = weighted_shared_gaussian_vm_nll_torch(
                m_va,
                L_va,
                logits_c,
                ld,
                kd,
                y_cal,
                yraw_cal[:, 2],
                w_cal,
                T_va=1.0,
                T_ang=1.0,
                kappa_max=kappa_max,
            )
            if wn < best_cal:
                best_cal = wn
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch {epoch+1}/{tcfg['max_epochs']} train_loss~{epoch_loss/n_batches:.4f} "
                f"cal_wNLL={wn:.4f} best={best_cal:.4f} patience={bad}/{patience}"
            )
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits_c, m_va, L_va, ld, kd = model(x_cal, cat_cal)
    ccal = cfg["calibration"]
    vm_temp = fit_hybrid_va_vm_temperatures(
        m_va.detach(),
        L_va.detach(),
        logits_c.detach(),
        ld.detach(),
        kd.detach(),
        y_cal,
        yraw_cal[:, 2],
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
        kappa_max=kappa_max,
    )
    train_sec_total = time.time() - t0

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "config": cfg,
            "target_means": train_a.target_means.tolist(),
            "target_stds": train_a.target_stds.tolist(),
            "numeric_feature_names": train_a.feature_names,
            "head_family": "shared_gaussian_vm_d",
            "temperature_T_va": vm_temp.T_va,
            "temperature_T_ang": vm_temp.T_ang,
            "kappa_max": kappa_max,
            "train_loop_seconds": train_loop_sec,
            "train_total_seconds_including_calibration": train_sec_total,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_va": vm_temp.T_va,
                "T_ang": vm_temp.T_ang,
                "note": "(v_ss,a): scale_tril *= sqrt(T_va); von Mises: kappa_cal = kappa / T_ang",
                "head_family": "shared_gaussian_vm_d",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "covariance_temperature_T_va": vm_temp.T_va,
        "angular_temperature_T_ang": vm_temp.T_ang,
    }


def train_stage_u_frozen_bounded_support(
    cfg: dict[str, Any],
    train_a: UArrays,
    val_a: UArrays,
    temp_cal_a: UArrays,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
    va_transform: PlayerSpecificBoundedLogitZScoreTransform,
    player_bounds: Any,
    debug_one_epoch: bool = False,
) -> dict[str, Any]:
    """
    Frozen-style shared Gaussian × von Mises MDN, but with player-specific bounded
    logit support for (v_ss_tilde, a_tilde) and a separate val/temp-cal split.
    """
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    rcfg = cfg.get("regularization", {})
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    n_mix = int(mcfg["n_mixture_components"])
    kappa_max = float(mcfg.get("kappa_max", 120.0))

    x_tr, cat_tr, y_tr, yraw_tr, w_tr = _arrays_to_tensors(train_a, device)
    x_val, cat_val, y_val, yraw_val, w_val = _arrays_to_tensors(val_a, device)
    x_temp, cat_temp, y_temp, yraw_temp, w_temp = _arrays_to_tensors(temp_cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = SharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_mixture=n_mix,
        hidden_width=mcfg["hidden_width"],
        n_hidden=mcfg["hidden_layers"],
        activation=mcfg.get("activation", "tanh"),
        dropout=tcfg.get("dropout", 0.0),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=kappa_max,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    batch_events = int(tcfg["batch_events"])
    max_draws = int(tcfg["max_draws_per_event"])
    max_epochs = 1 if debug_one_epoch else int(tcfg["max_epochs"])
    patience = int(tcfg["early_stopping_patience"])
    best_val = float("inf")
    bad = 0
    best_state = None
    curves: list[dict[str, Any]] = []
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
                ix.append(pool if len(pool) <= max_draws else rng.choice(pool, size=max_draws, replace=False))
            if not ix:
                continue
            bix = np.concatenate(ix, axis=0)
            bx = x_tr[bix]
            bc = {k: cat_tr[k][bix] for k in cat_tr}
            by = y_tr[bix]
            byraw = yraw_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            logits, mu_va, L_va, loc_d, kappa_d = model(bx, bc)
            lp = model.joint_log_prob(by, byraw[:, 2], logits, mu_va, L_va, loc_d, kappa_d)
            nll = -lp
            w_nll = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            reg, reg_terms = _explicit_l1_l2_regularization(model, rcfg)
            loss = w_nll + reg
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(w_nll.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches produced")

        model.eval()
        with torch.no_grad():
            logits_v, m_va, L_va, ld, kd = model(x_val, cat_val)
        val_wnll, _ = weighted_shared_gaussian_vm_nll_torch(
            m_va,
            L_va,
            logits_v,
            ld,
            kd,
            y_val,
            yraw_val[:, 2],
            w_val,
            T_va=1.0,
            T_ang=1.0,
            kappa_max=kappa_max,
        )
        if val_wnll < best_val:
            best_val = val_wnll
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        curves.append(
            {
                "epoch": epoch + 1,
                "train_weighted_nll_mean": epoch_loss / n_batches,
                "train_loss_mean": epoch_loss / n_batches,
                "val_select_weighted_nll": val_wnll,
                "regularization": reg_terms if "reg_terms" in locals() else {},
            }
        )
        print(
            f"epoch {epoch+1}/{max_epochs} train~{epoch_loss/n_batches:.4f} "
            f"val_select_wNLL={val_wnll:.4f} best={best_val:.4f} patience={bad}/{patience}"
        )
        if bad >= patience and not debug_one_epoch:
            print("Early stopping on val_select.")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0
    model.eval()
    with torch.no_grad():
        logits_t, m_va, L_va, ld, kd = model(x_temp, cat_temp)
    ccal = cfg["calibration"]
    vm_temp = fit_hybrid_va_vm_temperatures(
        m_va.detach(),
        L_va.detach(),
        logits_t.detach(),
        ld.detach(),
        kd.detach(),
        y_temp,
        yraw_temp[:, 2],
        w_temp,
        max_iter=int(ccal.get("max_iter", ccal.get("steps", 600))),
        lr=float(ccal.get("lr", 0.05)),
        kappa_max=kappa_max,
    )
    train_sec_total = time.time() - t0

    (out_dir / "target_transform.json").write_text(
        json.dumps(va_transform.state_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "player_target_bounds.json").write_text(
        json.dumps(player_bounds.state_dict(), indent=2), encoding="utf-8"
    )
    if curves:
        (out_dir / "training_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    (out_dir / "regularization_config.json").write_text(json.dumps(rcfg, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "config": cfg,
            "target_means": train_a.target_means.tolist(),
            "target_stds": train_a.target_stds.tolist(),
            "numeric_feature_names": train_a.feature_names,
            "head_family": FROZEN_BOUNDED_SUPPORT_HEAD,
            "temperature_T_va": vm_temp.T_va,
            "temperature_T_ang": vm_temp.T_ang,
            "kappa_max": kappa_max,
            "target_transform": va_transform.state_dict(),
            "player_target_bounds": player_bounds.state_dict(),
            "n_mixture_components": n_mix,
            "regularization_config": rcfg,
            "train_loop_seconds": train_loop_sec,
            "train_total_seconds_including_calibration": train_sec_total,
            "best_val_select_weighted_nll_pre_temp": best_val,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_va": vm_temp.T_va,
                "T_ang": vm_temp.T_ang,
                "calibration_split": "u_temp_cal",
                "head_family": FROZEN_BOUNDED_SUPPORT_HEAD,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_val_select_weighted_nll_pre_temp": best_val,
        "covariance_temperature_T_va": vm_temp.T_va,
        "angular_temperature_T_ang": vm_temp.T_ang,
        "training_curves": curves,
    }


def train_stage_u_bounded_hier_shared_gaussian_vonmises(
    cfg: dict[str, Any],
    train_a: UArrays,
    cal_a: UArrays,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
    va_transform: BoundedLogitZScoreTransform,
    debug_one_epoch: bool = False,
) -> dict[str, Any]:
    """
    Bounded logit-zscore (v_ss,a) + hierarchical player pooling + shared VM mixture on d.
    """
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    cvcfg = cfg.get("covariance", {})
    rcfg = cfg.get("regularization", {})
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    n_mix = _n_mixture_from_cfg(mcfg)
    kappa_max = float(mcfg.get("kappa_max", 120.0))
    hidden = int(mcfg.get("hidden_width", mcfg.get("hidden_dim", 64)))
    n_hidden = int(mcfg.get("hidden_layers", mcfg.get("depth", 2)))

    x_tr, cat_tr, y_tr, yraw_tr, w_tr, bid_tr = _arrays_to_tensors_bounded(train_a, device)
    x_cal, cat_cal, y_cal, yraw_cal, w_cal, bid_cal = _arrays_to_tensors_bounded(cal_a, device)
    cat_ord_tr = _ordinary_cat_dict(cat_tr)
    cat_ord_cal = _ordinary_cat_dict(cat_cal)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    n_batters = len(vocabs["batter_name"])
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in cfg["categorical_features"]}
    vocab_sizes_ord = {k: len(vocabs[k]) for k in ("pitch_type", "stand", "p_throws")}

    model = BoundedHierSharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        n_batters=n_batters,
        vocab_sizes=vocab_sizes_ord,
        embedding_dims=emb_dims,
        n_mixture=n_mix,
        hidden_width=hidden,
        n_hidden=n_hidden,
        activation=mcfg.get("activation", "tanh"),
        dropout=tcfg.get("dropout", 0.0),
        player_embedding_dim=int(mcfg.get("player_embedding_dim", 8)),
        player_effect_scale=float(mcfg.get("player_effect_scale", 1.0)),
        chol_eps=float(cvcfg.get("chol_eps", 1e-4)),
        diag_floor=float(cvcfg.get("diag_floor", 0.05)),
        diag_ceiling=float(cvcfg.get("diag_ceiling", 5.0)),
        use_diag_ceiling=bool(cvcfg.get("use_diag_ceiling", True)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=kappa_max,
    ).to(device)

    wd = float(rcfg.get("optimizer_weight_decay", tcfg.get("weight_decay", 3e-4)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=wd)

    batch_events = int(tcfg["batch_events"])
    max_draws = int(tcfg.get("max_draws_per_event", tcfg.get("draws_per_event", 16)))
    max_epochs = 1 if debug_one_epoch else int(tcfg["max_epochs"])
    patience = int(tcfg["early_stopping_patience"])
    grad_clip = float(tcfg.get("grad_clip", tcfg.get("grad_clip_norm", 5.0)))

    best_cal = float("inf")
    bad = 0
    best_state = None
    t0 = time.time()
    curves: list[dict[str, Any]] = []

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
            bx = x_tr[bix]
            bc = {k: cat_ord_tr[k][bix] for k in cat_ord_tr}
            bbid = bid_tr[bix]
            by = y_tr[bix]
            byraw = yraw_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            logits, mu_va, L_va, loc_d, kappa_d = model(bx, bc, bbid)
            lp = model.joint_log_prob(by, byraw[:, 2], logits, mu_va, L_va, loc_d, kappa_d)
            nll = -lp
            w_nll = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            reg, reg_terms = _bounded_hier_regularization(model, logits, rcfg)
            loss = w_nll + reg
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches produced")

        cal_wnll = float("inf")
        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                logits_c, m_va, L_va, ld, kd = model(x_cal, cat_ord_cal, bid_cal)
            cal_wnll, _ = weighted_shared_gaussian_vm_nll_torch(
                m_va,
                L_va,
                logits_c,
                ld,
                kd,
                y_cal,
                yraw_cal[:, 2],
                w_cal,
                T_va=1.0,
                T_ang=1.0,
                kappa_max=kappa_max,
            )
            if cal_wnll < best_cal:
                best_cal = cal_wnll
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            row = {
                "epoch": epoch + 1,
                "train_loss_mean": epoch_loss / n_batches,
                "calibration_weighted_nll": cal_wnll,
                **{f"reg_{k}": v for k, v in reg_terms.items()},
            }
            curves.append(row)
            print(
                f"epoch {epoch+1}/{max_epochs} train_loss~{epoch_loss/n_batches:.4f} "
                f"cal_wNLL={cal_wnll:.4f} best={best_cal:.4f} patience={bad}/{patience}"
            )
            if bad >= patience and not debug_one_epoch:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits_c, m_va, L_va, ld, kd = model(x_cal, cat_ord_cal, bid_cal)
    ccal = cfg["calibration"]
    vm_temp = fit_hybrid_va_vm_temperatures(
        m_va.detach(),
        L_va.detach(),
        logits_c.detach(),
        ld.detach(),
        kd.detach(),
        y_cal,
        yraw_cal[:, 2],
        w_cal,
        max_iter=int(ccal.get("max_iter", ccal.get("steps", 400))),
        lr=float(ccal.get("lr", 0.05)),
        kappa_max=kappa_max,
    )
    train_sec_total = time.time() - t0

    (out_dir / "target_transform.json").write_text(
        json.dumps(va_transform.state_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "regularization_config.json").write_text(
        json.dumps(rcfg, indent=2), encoding="utf-8"
    )
    import csv

    if curves:
        with (out_dir / "training_curves.csv").open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(curves)
        (out_dir / "training_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")

    ckpt_extra = {
        "model_state": model.state_dict(),
        "vocabs": vocabs,
        "config": cfg,
        "target_means": train_a.target_means.tolist(),
        "target_stds": train_a.target_stds.tolist(),
        "numeric_feature_names": train_a.feature_names,
        "head_family": BoundedHierSharedMixtureGaussianVonMisesUNet.HEAD_FAMILY,
        "temperature_T_va": vm_temp.T_va,
        "temperature_T_ang": vm_temp.T_ang,
        "kappa_max": kappa_max,
        "target_transform": va_transform.state_dict(),
        "player_embedding_dim": int(mcfg.get("player_embedding_dim", 8)),
        "player_effect_scale": float(mcfg.get("player_effect_scale", 1.0)),
        "n_mixture_components": n_mix,
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
    }
    torch.save(ckpt_extra, out_dir / "checkpoint.pt")
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_va": vm_temp.T_va,
                "T_ang": vm_temp.T_ang,
                "note": "(v_ss,a) transformed: scale_tril *= sqrt(T_va); von Mises: kappa_cal = kappa / T_ang",
                "head_family": BoundedHierSharedMixtureGaussianVonMisesUNet.HEAD_FAMILY,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "covariance_temperature_T_va": vm_temp.T_va,
        "angular_temperature_T_ang": vm_temp.T_ang,
        "training_curves": curves,
    }


def train_stage_u_bounded_player_support(
    cfg: dict[str, Any],
    train_a: UArrays,
    val_a: UArrays,
    temp_cal_a: UArrays,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
    va_transform: PlayerSpecificBoundedLogitZScoreTransform,
    player_bounds: Any,
    debug_one_epoch: bool = False,
) -> dict[str, Any]:
    """Train on u_train; early stop on u_val_select; temperatures on u_temp_cal only."""
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    rcfg = cfg.get("regularization", {})
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    kappa_max = float(mcfg.get("kappa_max", 120.0))
    n_mix = _n_mixture_from_cfg(mcfg)

    x_tr, cat_tr, y_tr, yraw_tr, w_tr, bid_tr = _arrays_to_tensors_bounded(train_a, device)
    x_val, cat_val, y_val, yraw_val, w_val, bid_val = _arrays_to_tensors_bounded(val_a, device)
    x_temp, cat_temp, y_temp, yraw_temp, w_temp, bid_temp = _arrays_to_tensors_bounded(
        temp_cal_a, device
    )
    cat_ord_tr = _ordinary_cat_dict(cat_tr)
    cat_ord_val = _ordinary_cat_dict(cat_val)
    cat_ord_temp = _ordinary_cat_dict(cat_temp)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])
    num_f = train_a.x_num.shape[1]

    model = build_bounded_player_support_model(cfg, num_f, vocabs, device)
    wd = float(rcfg.get("optimizer_weight_decay", tcfg.get("weight_decay", 3e-4)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=wd)

    batch_events = int(tcfg["batch_events"])
    max_draws = int(tcfg.get("max_draws_per_event", tcfg.get("draws_per_event", 16)))
    max_epochs = 1 if debug_one_epoch else int(tcfg["max_epochs"])
    patience = int(tcfg["early_stopping_patience"])
    grad_clip = float(tcfg.get("grad_clip", tcfg.get("grad_clip_norm", 5.0)))

    best_val = float("inf")
    bad = 0
    best_state = None
    t0 = time.time()
    curves: list[dict[str, Any]] = []

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
                ix.append(pool if len(pool) <= max_draws else rng.choice(pool, size=max_draws, replace=False))
            if not ix:
                continue
            bix = np.concatenate(ix, axis=0)
            bx, bc, bbid = x_tr[bix], {k: cat_ord_tr[k][bix] for k in cat_ord_tr}, bid_tr[bix]
            by, byraw, bw = y_tr[bix], yraw_tr[bix], w_tr[bix]
            opt.zero_grad()
            logits, mu_va, L_va, loc_d, kappa_d = model(bx, bc, bbid)
            lp = model.joint_log_prob(by, byraw[:, 2], logits, mu_va, L_va, loc_d, kappa_d)
            nll = -lp
            w_nll = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            reg, reg_terms = _bounded_hier_regularization(model, logits, rcfg)
            loss = w_nll + reg
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches produced")

        model.eval()
        with torch.no_grad():
            logits_v, m_va, L_va, ld, kd = model(x_val, cat_ord_val, bid_val)
        val_wnll, _ = weighted_shared_gaussian_vm_nll_torch(
            m_va, L_va, logits_v, ld, kd, y_val, yraw_val[:, 2], w_val,
            T_va=1.0, T_ang=1.0, kappa_max=kappa_max,
        )
        if val_wnll < best_val:
            best_val = val_wnll
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        curves.append(
            {"epoch": epoch + 1, "train_loss_mean": epoch_loss / n_batches, "val_select_weighted_nll": val_wnll}
        )
        print(
            f"epoch {epoch+1}/{max_epochs} train~{epoch_loss/n_batches:.4f} "
            f"val_select_wNLL={val_wnll:.4f} best={best_val:.4f} patience={bad}/{patience}"
        )
        if bad >= patience and not debug_one_epoch:
            print("Early stopping on val_select.")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_loop_sec = time.time() - t0
    model.eval()
    with torch.no_grad():
        logits_t, m_va, L_va, ld, kd = model(x_temp, cat_ord_temp, bid_temp)
    ccal = cfg["calibration"]
    vm_temp = fit_hybrid_va_vm_temperatures(
        m_va.detach(), L_va.detach(), logits_t.detach(), ld.detach(), kd.detach(),
        y_temp, yraw_temp[:, 2], w_temp,
        max_iter=int(ccal.get("max_iter", ccal.get("steps", 600))),
        lr=float(ccal.get("lr", 0.05)),
        kappa_max=kappa_max,
    )
    train_sec_total = time.time() - t0

    (out_dir / "target_transform.json").write_text(
        json.dumps(va_transform.state_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "player_target_bounds.json").write_text(
        json.dumps(player_bounds.state_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "regularization_config.json").write_text(json.dumps(rcfg, indent=2), encoding="utf-8")
    if curves:
        (out_dir / "training_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "config": cfg,
            "target_means": train_a.target_means.tolist(),
            "target_stds": train_a.target_stds.tolist(),
            "numeric_feature_names": train_a.feature_names,
            "head_family": BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet.HEAD_FAMILY,
            "temperature_T_va": vm_temp.T_va,
            "temperature_T_ang": vm_temp.T_ang,
            "kappa_max": kappa_max,
            "target_transform": va_transform.state_dict(),
            "player_target_bounds": player_bounds.state_dict(),
            "n_mixture_components": n_mix,
            "train_loop_seconds": train_loop_sec,
            "train_total_seconds_including_calibration": train_sec_total,
            "best_val_select_weighted_nll_pre_temp": best_val,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_va": vm_temp.T_va,
                "T_ang": vm_temp.T_ang,
                "calibration_split": "u_temp_cal",
                "head_family": BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet.HEAD_FAMILY,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_val_select_weighted_nll_pre_temp": best_val,
        "covariance_temperature_T_va": vm_temp.T_va,
        "angular_temperature_T_ang": vm_temp.T_ang,
        "training_curves": curves,
    }
