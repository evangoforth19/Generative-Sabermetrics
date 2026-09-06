"""Training loop for stage-z conditional full-covariance Gaussian mixture."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration_z import (
    _hybrid_scaled_L,
    fit_hybrid_native_temperature,
    fit_hybrid_trunc_ex_temperature,
    fit_mdn_temperature,
    trunc_ex_tempered_joint_log_prob,
)
from .data_u import index_by_event
from .data_z import ZArrays, ZArraysCircular, ZArraysNativeCircular, ZArraysNativeTruncEx
from .eval_z import weighted_mdn_nll_chol_torch
from .models_z import (
    ConditionalGaussianMixtureZ,
    ConditionalHybridGaussVonMisesMixtureZ,
    ConditionalHybridGaussVonMisesTruncExZ,
    mdn_hybrid_gauss_vm_log_prob,
    mdn_log_prob_chol,
)
from .schema import Z_TARGET_COLUMNS


def _to_tensors_train(arr: ZArrays | ZArraysCircular, device: torch.device):
    x = torch.from_numpy(arr.x_num).to(device)
    cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    y = torch.from_numpy(arr.y).to(device)
    w = torch.from_numpy(arr.w).to(device)
    return x, cat, y, w


def _to_tensors_train_native(arr: ZArraysNativeCircular, device: torch.device):
    x = torch.from_numpy(arr.x_num).to(device)
    cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    y_g = torch.from_numpy(arr.y_gauss).to(device)
    psi = torch.from_numpy(arr.psi_rad).to(device)
    theta = torch.from_numpy(arr.theta_rad).to(device)
    w = torch.from_numpy(arr.w).to(device)
    return x, cat, y_g, psi, theta, w


def _to_tensors_train_trunc_ex(arr: ZArraysNativeTruncEx, device: torch.device):
    x = torch.from_numpy(arr.x_num).to(device)
    cat = {k: torch.from_numpy(v).long().to(device) for k, v in arr.cat.items()}
    y_g = torch.from_numpy(arr.y_gauss).to(device)
    psi = torch.from_numpy(arr.psi_rad).to(device)
    ex = torch.from_numpy(arr.e_x).to(device)
    w = torch.from_numpy(arr.w).to(device)
    return x, cat, y_g, psi, ex, w


def _weighted_hybrid_nll(
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


def train_stage_z_mdn(
    cfg: dict[str, Any],
    train_a: ZArrays | ZArraysCircular,
    cal_a: ZArrays | ZArraysCircular,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    x_tr, cat_tr, y_tr, w_tr = _to_tensors_train(train_a, device)
    x_cal, cat_cal, y_cal, w_cal = _to_tensors_train(cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    z_dim = int(mcfg["z_dim"])
    n_mix = int(mcfg["n_components"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = ConditionalGaussianMixtureZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        z_dim=z_dim,
        n_components=n_mix,
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(tcfg.get("dropout", 0.0)),
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
            logits, mu, L = model(bx, bc)
            lp = mdn_log_prob_chol(by, logits, mu, L)
            nll = -lp
            loss = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches")

        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                lg_c, m_c, L_c = model(x_cal, cat_cal)
            wn, _ = weighted_mdn_nll_chol_torch(lg_c, m_c, L_c, y_cal, w_cal)
            if wn < best_cal:
                best_cal = wn
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch {epoch+1}/{tcfg['max_epochs']} train~{epoch_loss/n_batches:.4f} "
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
        lg_c, m_c, L_c = model(x_cal, cat_cal)
    ccal = cfg["calibration"]
    temp = fit_mdn_temperature(
        lg_c.detach(),
        m_c.detach(),
        L_c.detach(),
        y_cal,
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
    )
    train_sec_total = time.time() - t0

    ckpt_base: dict[str, Any] = {
        "model_state": model.state_dict(),
        "vocabs": vocabs,
        "config": cfg,
        "numeric_feature_names": train_a.feature_names,
        "temperature_T_mix": temp.T,
        "model_family": "full_cov_chol_gmm",
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
    }
    if isinstance(train_a, ZArraysCircular):
        ckpt_base["target_parameterization"] = "circular_6d"
        ckpt_base["circ_means"] = train_a.circ_means.tolist()
        ckpt_base["circ_stds"] = train_a.circ_stds.tolist()
        ckpt_base["circ_names"] = list(train_a.circ_names)
        ckpt_base["raw_target_order"] = list(Z_TARGET_COLUMNS)
    else:
        ckpt_base["target_means"] = train_a.target_means.tolist()
        ckpt_base["target_stds"] = train_a.target_stds.tolist()
        ckpt_base["target_parameterization"] = "linear_4d"
    torch.save(ckpt_base, out_dir / "checkpoint.pt")
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_mix": temp.T,
                "note": "Full-covariance mixture: Sigma_cal = T_mix * Sigma; L_cal = sqrt(T_mix) * L.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "temperature_T_mix": temp.T,
    }


def train_stage_z_hybrid_native(
    cfg: dict[str, Any],
    train_a: ZArraysNativeCircular,
    cal_a: ZArraysNativeCircular,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Train hybrid Gaussian (x, e_y*) + von Mises (psi, theta) mixture with shared weights."""
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    x_tr, cat_tr, yg_tr, psi_tr, th_tr, w_tr = _to_tensors_train_native(train_a, device)
    x_cal, cat_cal, yg_cal, psi_cal, th_cal, w_cal = _to_tensors_train_native(cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    n_mix = int(mcfg["n_components"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = ConditionalHybridGaussVonMisesMixtureZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=n_mix,
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(tcfg.get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
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
            byg = yg_tr[bix]
            bp = psi_tr[bix]
            bt = th_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            lg, mg, Lg, mp, kp, mt, kt = model(bx, bc)
            lp = mdn_hybrid_gauss_vm_log_prob(byg, bp, bt, lg, mg, Lg, mp, kp, mt, kt)
            nll = -lp
            loss = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches")

        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                lg_c, mg_c, Lg_c, mp_c, kp_c, mt_c, kt_c = model(x_cal, cat_cal)
            wn, _ = _weighted_hybrid_nll(lg_c, mg_c, Lg_c, mp_c, kp_c, mt_c, kt_c, yg_cal, psi_cal, th_cal, w_cal)
            if wn < best_cal:
                best_cal = wn
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch {epoch+1}/{tcfg['max_epochs']} train~{epoch_loss/n_batches:.4f} "
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
        lg_c, mg_c, Lg_c, mp_c, kp_c, mt_c, kt_c = model(x_cal, cat_cal)
    ccal = cfg["calibration"]
    htemp = fit_hybrid_native_temperature(
        lg_c.detach(),
        mg_c.detach(),
        Lg_c.detach(),
        mp_c.detach(),
        kp_c.detach(),
        mt_c.detach(),
        kt_c.detach(),
        yg_cal,
        psi_cal,
        th_cal,
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
        anisotropic_gaussian=bool(ccal.get("anisotropic_gaussian", False)),
    )
    train_sec_total = time.time() - t0

    ckpt_base: dict[str, Any] = {
        "model_state": model.state_dict(),
        "vocabs": vocabs,
        "config": cfg,
        "numeric_feature_names": train_a.feature_names,
        "temperature_T_gauss": htemp.T_gauss,
        "temperature_T_gauss_x": htemp.T_gauss_x,
        "temperature_T_gauss_y": htemp.T_gauss_y,
        "temperature_T_ang": htemp.T_ang,
        "model_family": "hybrid_gauss_vonmises_shared",
        "gauss_means": train_a.gauss_means.tolist(),
        "gauss_stds": train_a.gauss_stds.tolist(),
        "target_parameterization": "native_circular_vm",
        "raw_target_order": list(Z_TARGET_COLUMNS),
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
    }
    torch.save(ckpt_base, out_dir / "checkpoint.pt")
    temp_note = (
        "Gaussian: row i of L scaled by sqrt(T_gauss_i) for x and e_y* "
        "(Σ_cal = D Σ D); T_gauss is geometric mean of T_x,T_y. "
        "Angular: kappa_cal = kappa / T_ang for ψ and θ."
    )
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_gauss": htemp.T_gauss,
                "T_gauss_x": htemp.T_gauss_x,
                "T_gauss_y": htemp.T_gauss_y,
                "T_ang": htemp.T_ang,
                "note": temp_note,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "temperature_T_gauss": htemp.T_gauss,
        "temperature_T_gauss_x": htemp.T_gauss_x,
        "temperature_T_gauss_y": htemp.T_gauss_y,
        "temperature_T_ang": htemp.T_ang,
    }


def _weighted_trunc_ex_joint_nll(
    model: ConditionalHybridGaussVonMisesTruncExZ,
    x: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_g: torch.Tensor,
    psi: torch.Tensor,
    ex: torch.Tensor,
    w: torch.Tensor,
) -> tuple[float, float]:
    logits, mu_g, L_g, mu_psi, kappa_psi, h = model(x, cat)
    lp = model.log_prob_joint(y_g, psi, ex, logits, mu_g, L_g, mu_psi, kappa_psi, h)
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    return float((nll * w).sum().item() / wsum), float(nll.mean().item())


def weighted_trunc_ex_joint_nll_tempered(
    model: ConditionalHybridGaussVonMisesTruncExZ,
    x: torch.Tensor,
    cat: dict[str, torch.Tensor],
    y_g: torch.Tensor,
    psi: torch.Tensor,
    ex: torch.Tensor,
    w: torch.Tensor,
    *,
    T_gauss_x: float,
    T_gauss_y: float,
    T_ang: float,
) -> tuple[float, float]:
    """Weighted / mean joint NLL with post-hoc Gaussian + ψ temperatures (e_x head unchanged)."""
    logits, mu_g, L_g, mu_psi, kappa_psi, h = model(x, cat)
    mu_ex, sig_ex = model.ex_params(h, y_g, psi)
    dev, dt = y_g.device, y_g.dtype
    tx = torch.tensor(T_gauss_x, device=dev, dtype=dt)
    ty = torch.tensor(T_gauss_y, device=dev, dtype=dt)
    ta = torch.tensor(T_ang, device=dev, dtype=dt)
    lp = trunc_ex_tempered_joint_log_prob(
        y_g,
        psi,
        ex,
        logits,
        mu_g,
        L_g,
        mu_psi,
        kappa_psi,
        mu_ex,
        sig_ex,
        T_gauss_x=tx,
        T_gauss_y=ty,
        T_ang=ta,
    )
    nll = -lp
    wsum = float(w.sum().clamp_min(1e-12).item())
    return float((nll * w).sum().item() / wsum), float(nll.mean().item())


def train_stage_z_hybrid_trunc_ex(
    cfg: dict[str, Any],
    train_a: ZArraysNativeTruncEx,
    cal_a: ZArraysNativeTruncEx,
    vocabs: dict[str, dict[str, int]],
    *,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Train hybrid Gaussian (x, e_y*) + von Mises(psi) + conditional truncated-normal e_x."""
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    x_tr, cat_tr, yg_tr, psi_tr, ex_tr, w_tr = _to_tensors_train_trunc_ex(train_a, device)
    x_cal, cat_cal, yg_cal, psi_cal, ex_cal, w_cal = _to_tensors_train_trunc_ex(cal_a, device)

    idx_by_event = index_by_event(train_a.event_id)
    event_list = list(idx_by_event.keys())
    rng = np.random.default_rng(tcfg["seed"])

    num_f = train_a.x_num.shape[1]
    n_mix = int(mcfg["n_components"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}

    model = ConditionalHybridGaussVonMisesTruncExZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=n_mix,
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(tcfg.get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        sigma_floor=float(mcfg.get("sigma_floor", 1e-3)),
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
            byg = yg_tr[bix]
            bp = psi_tr[bix]
            bex = ex_tr[bix]
            bw = w_tr[bix]

            opt.zero_grad()
            lg, mg, Lg, mp, kp, h = model(bx, bc)
            lp = model.log_prob_joint(byg, bp, bex, lg, mg, Lg, mp, kp, h)
            nll = -lp
            loss = (nll * bw).sum() / bw.sum().clamp_min(1e-12)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches")

        if (epoch + 1) % int(tcfg.get("eval_calibration_every", 1)) == 0:
            model.eval()
            with torch.no_grad():
                wn, _ = _weighted_trunc_ex_joint_nll(model, x_cal, cat_cal, yg_cal, psi_cal, ex_cal, w_cal)
            if wn < best_cal:
                best_cal = wn
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch {epoch+1}/{tcfg['max_epochs']} train~{epoch_loss/n_batches:.4f} "
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
        lg_c, mg_c, Lg_c, mp_c, kp_c, h_c = model(x_cal, cat_cal)
        mu_ex_c, sig_ex_c = model.ex_params(h_c, yg_cal, psi_cal)
    ccal = cfg["calibration"]
    htemp = fit_hybrid_trunc_ex_temperature(
        lg_c.detach(),
        mg_c.detach(),
        Lg_c.detach(),
        mp_c.detach(),
        kp_c.detach(),
        mu_ex_c.detach(),
        sig_ex_c.detach(),
        yg_cal,
        psi_cal,
        ex_cal,
        w_cal,
        max_iter=int(ccal["max_iter"]),
        lr=float(ccal.get("lr", 0.05)),
        anisotropic_gaussian=bool(ccal.get("anisotropic_gaussian", False)),
    )
    train_sec_total = time.time() - t0
    targets = list(cfg["targets"])

    ckpt_base: dict[str, Any] = {
        "model_state": model.state_dict(),
        "vocabs": vocabs,
        "config": cfg,
        "numeric_feature_names": train_a.feature_names,
        "temperature_T_gauss": htemp.T_gauss,
        "temperature_T_gauss_x": htemp.T_gauss_x,
        "temperature_T_gauss_y": htemp.T_gauss_y,
        "temperature_T_ang": htemp.T_ang,
        "model_family": "hybrid_gauss_vonmises_trunc_ex",
        "gauss_means": train_a.gauss_means.tolist(),
        "gauss_stds": train_a.gauss_stds.tolist(),
        "target_parameterization": "native_trunc_ex_vm",
        "raw_target_order": targets,
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
    }
    torch.save(ckpt_base, out_dir / "checkpoint.pt")
    temp_note = (
        "native_trunc_ex_vm: Gaussian block same as circular (Σ via row-scaled Cholesky; "
        "T_gauss geom mean of T_x,T_y). κ_cal = κ/T_ang for ψ only. "
        "Truncated-normal e_x factor uses learned σ (not separately temperature-scaled)."
    )
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "T_gauss": htemp.T_gauss,
                "T_gauss_x": htemp.T_gauss_x,
                "T_gauss_y": htemp.T_gauss_y,
                "T_ang": htemp.T_ang,
                "note": temp_note,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "train_loop_seconds": train_loop_sec,
        "train_total_seconds_including_calibration": train_sec_total,
        "best_calibration_weighted_nll_pre_temp": best_cal,
        "temperature_T_gauss": htemp.T_gauss,
        "temperature_T_gauss_x": htemp.T_gauss_x,
        "temperature_T_gauss_y": htemp.T_gauss_y,
        "temperature_T_ang": htemp.T_ang,
    }
