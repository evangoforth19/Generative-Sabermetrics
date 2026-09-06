"""Smoke tests for branch_autoreg_bounded_hybrid_mdn_z."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_PKG = _ROOT / "sbi_forward_sim"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbi_forward_sim.src.models_z import BranchAutoregBoundedHybridMDNZ  # noqa: E402
from sbi_forward_sim.src.target_transforms_z import (  # noqa: E402
    BoundedLogitZscoreState,
    fit_bounded_logit_zscore_state,
    model_to_raw_coords,
    raw_to_model_coords,
)


def test_bounded_transforms_roundtrip():
    st = BoundedLogitZscoreState(0.1, 1.2, -0.2, 0.9, 0.0, 1.1, 1e-5, "test", False, 0.0, 1.0, 0.0, 0.6, 0.6)
    x_lo = np.array([20.0])
    x_hi = np.array([34.0])
    x, ey, ex = 28.5, 0.42, 0.15
    rx, ry, rex, _ = raw_to_model_coords(x, ey, ex, x_lower=x_lo, x_upper=x_hi, state=st)
    x2, ey2, ex2 = model_to_raw_coords(rx, ry, rex, x_lower=x_lo, x_upper=x_hi, state=st)
    assert abs(x2[0] - x) < 1e-4
    assert abs(ey2[0] - ey) < 1e-4
    assert abs(ex2[0] - ex) < 1e-4
    assert 0 <= ey2[0] <= 1
    assert 0 <= ex2[0] <= 0.6


def test_model_forward_backward():
    B, F, K = 8, 16, 5
    model = BranchAutoregBoundedHybridMDNZ(
        input_dim=F,
        vocab_sizes={"batter_name": 10, "p_throws": 3, "pitch_type": 5, "stand": 2},
        embedding_dims={"batter_name": 4, "p_throws": 2, "pitch_type": 2, "stand": 2},
        n_components=K,
        hidden_width=32,
        n_hidden=2,
    )
    x = torch.randn(B, F)
    cat = {
        "batter_name": torch.randint(0, 10, (B,)),
        "p_throws": torch.randint(0, 3, (B,)),
        "pitch_type": torch.randint(0, 5, (B,)),
        "stand": torch.randint(0, 2, (B,)),
    }
    r_xy = torch.randn(B, 2)
    psi = torch.randn(B) * 0.5
    r_ex = torch.randn(B)
    h = model.trunk(x, cat)
    logits, mu, L = model.mixture_params(h)
    lp = model.log_prob_model_space(r_xy, psi, r_ex, logits, mu, L, h)
    assert torch.isfinite(lp).all()
    loss = -lp.mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_psi_head_depends_on_rxy():
    model = BranchAutoregBoundedHybridMDNZ(
        input_dim=8,
        vocab_sizes={"batter_name": 5},
        embedding_dims={"batter_name": 4},
        n_components=3,
        hidden_width=16,
        n_hidden=1,
    )
    x = torch.randn(2, 8)
    cat = {"batter_name": torch.tensor([0, 1])}
    h = model.trunk(x, cat)
    r1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    r2 = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
    m1, k1 = model.psi_params_all_branches(h, r1)
    m2, k2 = model.psi_params_all_branches(h, r2)
    assert not torch.allclose(m1, m2)
    assert not torch.allclose(k1, k2)


def test_old_trunc_ex_resume_eval():
    run = _PKG / "outputs" / "p_z_given_u_g" / "20260501_055904Z"
    if not (run / "checkpoint.pt").is_file():
        return
    script = _PKG / "scripts" / "train_p_z_given_u_g_native_trunc_ex.py"
    subprocess.run(
        [sys.executable, str(script), "--resume-eval", str(run)],
        cwd=str(_ROOT),
        check=True,
        timeout=600,
    )
