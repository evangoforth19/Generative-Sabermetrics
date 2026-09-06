"""Smoke tests for bounded player-support stage-u path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbi_forward_sim.src.player_target_bounds_u import (  # noqa: E402
    smallest_covering_arc_deg,
    fit_player_target_bounds_train_only,
)
from sbi_forward_sim.src.target_transforms_u import PlayerSpecificBoundedLogitZScoreTransform  # noqa: E402
from sbi_forward_sim.src.models_u import BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet  # noqa: E402


def test_circular_arc_wraps_zero():
    arc = smallest_covering_arc_deg(np.array([350.0, 355.0, 5.0, 10.0]))
    assert arc.wraps_zero or arc.arc_width_deg > 180
    assert arc.contains_deg(0.0)
    assert not arc.contains_deg(180.0)


def test_circular_arc_compact():
    arc = smallest_covering_arc_deg(np.array([30.0, 40.0, 50.0]))
    assert not arc.wraps_zero
    assert arc.contains_deg(35.0)
    assert not arc.contains_deg(10.0)


def test_player_transform_roundtrip():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "batter_name": ["alice"] * 100 + ["bob"] * 100,
            "v_ss_tilde": rng.uniform(40, 80, n),
            "a_tilde": rng.uniform(-20, 20, n),
            "d_tilde": rng.uniform(0, 360, n),
            "event_id": np.arange(n),
            "combined_training_weight": np.ones(n),
        }
    )
    cfg = {
        "bounds": {
            "player_specific": {
                "min_events_for_player_bounds": 1,
                "min_rows_for_player_bounds": 10,
                "attack_angle_global_cap": [-45.0, 45.0],
                "padding": {"v_ss_tilde_abs": 0.0, "a_tilde_abs": 0.0, "d_tilde_deg": 0.0},
            },
            "global_sanity_caps": {"v_ss_tilde": [20.0, 110.0], "a_tilde": [-45.0, 45.0]},
            "d_tilde_support": {"full_circle_threshold_deg": 350.0},
        }
    }
    vocabs = {"batter_name": {"<UNK>": 0, "alice": 1, "bob": 2}}
    pb = fit_player_target_bounds_train_only(df, cfg, vocabs=vocabs)
    tr = PlayerSpecificBoundedLogitZScoreTransform(pb)
    pids = np.array([1] * 100 + [2] * 100)
    raw = df[["v_ss_tilde", "a_tilde"]].to_numpy(dtype=np.float64)
    tr.fit(df, pids)
    yt = tr.transform(raw, pids)
    inv = tr.inverse(yt, pids)
    assert np.allclose(raw, inv, rtol=0.05, atol=2.0)
    for j in range(2):
        L, U = pb.va_lower_upper(pids, j)
        assert np.all(inv[:, j] >= L - 1e-3)
        assert np.all(inv[:, j] <= U + 1e-3)


def test_model_forward_backward():
    B, F, K = 32, 13, 6
    model = BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet(
        input_dim=F,
        n_batters=5,
        vocab_sizes={"pitch_type": 10, "stand": 2, "p_throws": 3},
        embedding_dims={"pitch_type": 8, "stand": 4, "p_throws": 4},
        n_mixture=K,
    )
    x = torch.randn(B, F)
    cat = {
        "pitch_type": torch.randint(0, 10, (B,)),
        "stand": torch.randint(0, 2, (B,)),
        "p_throws": torch.randint(0, 3, (B,)),
    }
    bid = torch.randint(0, 5, (B,))
    y = torch.randn(B, 3)
    d = torch.randn(B) * 30 + 180
    logits, mu, L, loc, kap = model(x, cat, bid)
    lp = model.joint_log_prob(y, d, logits, mu, L, loc, kap)
    loss = -lp.mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_player_bounds_quantile_and_global_d_cap():
    import pandas as pd

    df = pd.DataFrame(
        {
            "batter_name": ["alice"] * 100,
            "v_ss_tilde": np.r_[np.linspace(50.0, 70.0, 98), 10.0, 120.0],
            "a_tilde": np.r_[np.linspace(-10.0, 20.0, 98), -90.0, 350.0],
            "d_tilde": np.linspace(-60.0, 60.0, 100),
            "event_id": np.arange(100),
            "combined_training_weight": np.ones(100),
        }
    )
    cfg = {
        "bounds": {
            "player_specific": {
                "min_events_for_player_bounds": 1,
                "min_rows_for_player_bounds": 10,
                "quantile_bounds": {"enabled": True, "lower": 0.01, "upper": 0.99},
                "use_global_d_tilde_cap": True,
                "d_tilde_global_cap": [-45.0, 45.0],
                "padding": {"v_ss_tilde_abs": 0.0, "a_tilde_abs": 0.0, "d_tilde_deg": 0.0},
            },
            "global_sanity_caps": {"v_ss_tilde": [20.0, 110.0]},
            "d_tilde_support": {"full_circle_threshold_deg": 350.0},
        }
    }
    vocabs = {"batter_name": {"<UNK>": 0, "alice": 1}}
    pb = fit_player_target_bounds_train_only(df, cfg, vocabs=vocabs)
    v_lo, v_hi = pb.va_bounds["v_ss_tilde"][1]
    a_lo, a_hi = pb.va_bounds["a_tilde"][1]
    assert v_lo > 10.0
    assert v_hi < 120.0
    assert a_lo > -90.0
    assert a_hi < 350.0
    arc = pb.get_d_arc(1)
    assert arc.contains_deg(-45.0)
    assert arc.contains_deg(45.0)
    assert not arc.contains_deg(60.0)
