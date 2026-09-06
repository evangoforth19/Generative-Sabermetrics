#!/usr/bin/env python3
"""
Bootstrap real BIP contexts per hitter -> stage-u x stage-z x physics decoder -> (EV, LA, SA)
-> LightGBM xwOBAcon surrogate; accumulate until convergence; save surfaces + value samples.

Stage-z: supports legacy circular checkpoints (sampled ``theta_deg``) and
``hybrid_gauss_vonmises_trunc_ex`` (sampled ``e_x`` + analytic decoder for LA/θ).

Requires:
  - ``MCMC 2/sbi_forward_sim`` checkpoints (stage-u, stage-z) and frozen feature contract.
  - ``data/statcast_pybaseball/twelve_hitters_bip_full_statcast.parquet`` (or compatible BIP table).
  - ``outputs/batted_ball_value_surface`` LGBM bundle.

Run with ``MCMC 2/.venv_prod/bin/python`` (or any env where ``arviz`` is installed): the physics decoder
imports ``run_mcmc_posterior_bank``. Set ``PYTHONPATH=MCMC 2:<repo>/outputs/batted_ball_value_surface``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MMC2_ROOT = REPO_ROOT / "MCMC 2"
if str(MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(MMC2_ROOT))

from sbi_forward_sim.src.data_u import (  # noqa: E402
    load_all_u_data,
    load_standardization_stats,
    merge_player_constants,
)
from sbi_forward_sim.src.data_z import load_all_z_data_native_circular, load_all_z_data_native_trunc_ex  # noqa: E402
from sbi_forward_sim.src.feature_engineering import add_spin_axis_trig, z_count_from_balls_strikes  # noqa: E402
from sbi_forward_sim.src.feature_contract_z import load_feature_contract_z  # noqa: E402
from sbi_forward_sim.src.heldout_forward_inference import (  # noqa: E402
    u_encoder_inputs_from_row,
    z_base_x_num_from_row,
    z_categoricals_from_row,
)
from sbi_forward_sim.src.models_u import SharedMixtureGaussianVonMisesUNet  # noqa: E402
from sbi_forward_sim.src.models_z import (  # noqa: E402
    ConditionalHybridGaussVonMisesMixtureZ,
    ConditionalHybridGaussVonMisesTruncExZ,
)
from sbi_forward_sim.src.physics_decoder import decode_bip_batch  # noqa: E402
from sbi_forward_sim.src.physics_calibration import CONTEXT_COLUMNS, EVPhysicsCalibrator  # noqa: E402
from sbi_forward_sim.src.physics_decoder_contract import (  # noqa: E402
    DECODER_G_NUMERIC_REQUIRED,
    DECODER_G_OPTIONAL,
    DECODER_P_REQUIRED,
    DecoderInputs,
    DecoderInputError,
    validate_decoder_context_g,
)
from sbi_forward_sim.src.pipeline_heldout_sample import (  # noqa: E402
    patch_z_x_num_with_u,
    sample_u_shared_gaussian_vm,
    sample_z_hybrid_native_batched,
    sample_z_hybrid_trunc_ex_batched,
)
from sbi_forward_sim.src.spray_angle_bounds import (  # noqa: E402
    SPRAY_ANGLE_DEG_LIMITS,
    filter_bip_model_eligible,
    spray_angle_in_support,
)

VALUE_MODEL_ROOT = REPO_ROOT / "outputs" / "batted_ball_value_surface"
if str(VALUE_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(VALUE_MODEL_ROOT))

from batted_ball_value_model import (  # noqa: E402
    HOME_X,
    HOME_Y,
    load_batted_ball_value_model,
    predict_xwobacon3d,
)


def _build_bootstrap_context_pool(
    context_master: pd.DataFrame,
    *,
    players: list[str] | None = None,
) -> pd.DataFrame:
    """``bip_model``-eligible contexts with spray angle in [-45°, 45°]."""
    pool = filter_bip_model_eligible(context_master, limits=SPRAY_ANGLE_DEG_LIMITS)
    pool["batter_name"] = pool["batter_name"].astype(str).str.strip()
    if players is not None:
        pset = set(players)
        pool = pool.loc[pool["batter_name"].isin(pset)].copy()
    return pool.reset_index(drop=True)


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_u_model(cfg: dict, vocabs: dict, device: torch.device) -> SharedMixtureGaussianVonMisesUNet:
    mcfg = cfg["model"]
    num_f = len(cfg["numeric_features"]) + len(cfg["player_constant_features"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return SharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_mixture=int(mcfg["n_mixture_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=float(mcfg.get("kappa_max", 120.0)),
    ).to(device)


def _build_z_model(
    cfg: dict,
    vocabs: dict,
    device: torch.device,
    *,
    feature_contract: dict[str, Any] | None = None,
    z_project_root: Path | None = None,
) -> ConditionalHybridGaussVonMisesMixtureZ:
    mcfg = cfg["model"]
    zroot = z_project_root or (MMC2_ROOT / "sbi_forward_sim")
    fc = feature_contract or load_feature_contract_z(cfg, zroot)
    num_f = len(fc["x_numeric_zscore_column_order"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesMixtureZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=int(mcfg["n_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
    ).to(device)


def _build_z_model_trunc_ex(
    cfg: dict,
    vocabs: dict,
    device: torch.device,
    *,
    feature_contract: dict[str, Any] | None = None,
    z_project_root: Path | None = None,
) -> ConditionalHybridGaussVonMisesTruncExZ:
    mcfg = cfg["model"]
    zroot = z_project_root or (MMC2_ROOT / "sbi_forward_sim")
    fc = feature_contract or load_feature_contract_z(cfg, zroot)
    num_f = len(fc["x_numeric_zscore_column_order"])
    vocab_sizes = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb_dims = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesTruncExZ(
        input_dim=num_f,
        vocab_sizes=vocab_sizes,
        embedding_dims=emb_dims,
        n_components=int(mcfg["n_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        sigma_floor=float(mcfg.get("sigma_floor", 1e-3)),
    ).to(device)


def _prepare_stats_u(project_root: Path, u_cfg: dict[str, Any], master: pd.DataFrame) -> dict[str, Any]:
    stats_u = copy.deepcopy(load_standardization_stats(project_root / u_cfg["paths"]["standardization_stats"]))
    ut = pd.read_parquet(project_root / u_cfg["paths"]["u_train"])
    ut = merge_player_constants(ut, master)
    for c in u_cfg["player_constant_features"]:
        if c not in stats_u:
            v = pd.to_numeric(ut[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats_u[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
    return stats_u


def _prepare_stats_z(
    project_root: Path, z_cfg: dict[str, Any], z_fc: dict[str, Any], master: pd.DataFrame
) -> dict[str, Any]:
    stats = copy.deepcopy(load_standardization_stats(project_root / z_cfg["paths"]["standardization_stats"]))
    zt = pd.read_parquet(project_root / z_cfg["paths"]["z_train"])
    zt = merge_player_constants(zt, master)
    for t in z_fc["targets"]:
        if t not in stats:
            v = pd.to_numeric(zt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
    for c in z_fc["player_constant_features"]:
        if c not in stats:
            v = pd.to_numeric(zt[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
    return stats


def _g_dict_from_row(row: pd.Series) -> dict[str, Any]:
    g: dict[str, Any] = {}
    for k in DECODER_G_NUMERIC_REQUIRED:
        if k not in row.index:
            raise DecoderInputError(f"row missing decoder g key {k!r}")
        g[k] = float(row[k])
    has_deg = "spin_axis_deg" in row.index and pd.notna(row.get("spin_axis_deg"))
    if has_deg:
        g["spin_axis_deg"] = float(row["spin_axis_deg"])
    else:
        for k in ("spin_axis_sin", "spin_axis_cos"):
            if k not in row.index:
                raise DecoderInputError(f"missing {k} for spin bundle")
            g[k] = float(row[k])
    for ok in DECODER_G_OPTIONAL:
        if ok in row.index and pd.notna(row.get(ok)):
            g[ok] = float(row[ok])
    validate_decoder_context_g(g)
    return g


def _p_dict_from_row(row: pd.Series) -> dict[str, float]:
    p: dict[str, float] = {}
    for k in DECODER_P_REQUIRED:
        if k not in row.index or pd.isna(row.get(k)):
            raise DecoderInputError(f"missing player constant {k!r}")
        p[k] = float(row[k])
    return p


def _spray_angle_deg_series(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    x = pd.to_numeric(hc_x, errors="coerce")
    y = pd.to_numeric(hc_y, errors="coerce")
    return np.degrees(np.arctan2(x - HOME_X, HOME_Y - y))


def statcast_row_to_model_row(
    stat: pd.Series,
    *,
    event_id: int,
    p_const: dict[str, float],
) -> pd.Series:
    """Single BIP Statcast row -> merged row with z_count, spin trig, player constants."""
    df = pd.DataFrame([stat.to_dict()])
    if "z_count" in df.columns:
        df = df.drop(columns=["z_count"])
    df = z_count_from_balls_strikes(df, table_name="bootstrap_bip")
    df = add_spin_axis_trig(df, axis_col="spin_axis")
    for k, v in p_const.items():
        df[k] = v
    df["event_id"] = int(event_id)
    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].astype(str).str.strip()
    if "stand" in df.columns:
        df["stand"] = df["stand"].astype(str).str.strip()
    if "p_throws" in df.columns:
        df["p_throws"] = df["p_throws"].astype(str).str.strip()
    if "batter_name" in df.columns:
        df["batter_name"] = df["batter_name"].astype(str).str.strip()
    return pd.Series(df.iloc[0])


def _safe_dir_name(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip().lower())
    return s[:120] or "unknown"


_GROUP_MAP_BT: dict[str, list[str]] = {
    "4F": ["FF", "FA"],
    "2F": ["SI", "FT"],
    "CF": ["FC"],
    "S": ["SL", "ST"],
    "C": ["CU", "KC", "CS"],
    "CH": ["CH", "FS"],
}


def _pitch_type_to_group_bootstrap() -> dict[str, str]:
    o: dict[str, str] = {}
    for g, pts in _GROUP_MAP_BT.items():
        for pt in pts:
            o[pt.upper()] = g
    return o


_PT_TO_GROUP_BOOT = _pitch_type_to_group_bootstrap()


def _bootstrap_draws_df_for_calibration(
    row: pd.Series,
    ev_dec: np.ndarray,
    la: np.ndarray,
    sa: np.ndarray,
    calibrator: EVPhysicsCalibrator,
) -> pd.DataFrame:
    """Build draw table for EVPhysicsCalibrator from one Statcast context row + decoder samples."""
    n = int(ev_dec.shape[0])
    if n == 0:
        return pd.DataFrame()
    eid = int(row["event_id"])
    evc = calibrator.ev_col
    df = pd.DataFrame(
        {
            calibrator.group_col: np.full(n, eid, dtype=np.int64),
            evc: ev_dec.astype(np.float64),
            "LA": la.astype(np.float64),
            "SA": sa.astype(np.float64),
            "draw_idx": np.arange(n, dtype=np.int32),
        }
    )
    if "pitch_type" in row.index and pd.notna(row.get("pitch_type")):
        pt = str(row["pitch_type"]).strip().upper()
        df["pitch_type"] = row["pitch_type"]
        pg = _PT_TO_GROUP_BOOT.get(pt)
        if pg is not None:
            df["pitch_group"] = pg
    if "pitch_group" in row.index and "pitch_group" not in df.columns and pd.notna(row.get("pitch_group")):
        df["pitch_group"] = row["pitch_group"]
    for c in CONTEXT_COLUMNS:
        if c in ("pitch_group", "pitch_type"):
            continue
        if c in row.index and c not in df.columns:
            df[c] = row.get(c)
    if "spin_rate" not in df.columns and "release_spin_rate" in row.index:
        df["spin_rate"] = pd.to_numeric(row.get("release_spin_rate"), errors="coerce")
    if "pitcher_hand" not in df.columns and "p_throws" in row.index:
        df["pitcher_hand"] = row["p_throws"]
    if "batter_hand" not in df.columns and "stand" in row.index:
        df["batter_hand"] = row["stand"]
    return df


def convergence_ok(
    values: np.ndarray,
    *,
    min_n: int,
    rel_mean_tol: float,
    ks_max: float,
    frac: float,
) -> bool:
    """Heuristic: stable mean between early/late windows + small KS on value distribution."""
    from scipy.stats import ks_2samp

    v = np.asarray(values, dtype=np.float64).ravel()
    n = len(v)
    if n < min_n:
        return False
    w = max(500, int(n * frac))
    w = min(w, n // 2)
    if w < 100:
        return False
    early, late = v[:w], v[-w:]
    if not np.isfinite(early).all() or not np.isfinite(late).all():
        return False
    mu_e, mu_l = float(np.mean(early)), float(np.mean(late))
    sig = float(np.std(v))
    if not np.isfinite(sig) or sig < 1e-6:
        return abs(mu_l - mu_e) < rel_mean_tol
    rel = abs(mu_l - mu_e) / sig
    if rel > rel_mean_tol:
        return False
    a = v[: max(200, n // 5)]
    b = v[-max(200, n // 5) :]
    ks = float(ks_2samp(a, b).statistic)
    return ks <= ks_max


def sample_decode_value_batch(
    row: pd.Series,
    *,
    event_id: int,
    batter: str,
    n_samples: int,
    u_model,
    z_model,
    u_cfg: dict,
    stats_u: dict,
    u_vocabs: dict,
    stats_z: dict,
    z_vocabs: dict,
    z_num_order: list[str],
    u_cols_tpl: tuple[str, str, str],
    device: torch.device,
    T_va: float,
    T_ang_u: float,
    kappa_max_u: float,
    T_gauss_z: float,
    T_ang_z: float,
    T_gx_f: float | None,
    T_gy_f: float | None,
    tm_u: torch.Tensor,
    ts_u: torch.Tensor,
    gm_z: torch.Tensor,
    gs_z: torch.Tensor,
    g_gen_u: torch.Generator,
    g_gen_z: torch.Generator,
    model_bundle: dict,
    decode_chunk: int,
    z_trunc_ex: bool,
    physics_calibrator: EVPhysicsCalibrator | None = None,
    d_tilde_deg_limits: tuple[float, float] | None = None,
    enforce_sa_limits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    One context row, ``n_samples`` joint draws. Returns (ev, la, sa, xwoba) for **admissible** only
    (variable length), and count attempted.
    """
    g_dec = _g_dict_from_row(row)
    p_dec = _p_dict_from_row(row)
    x_u_np, cat_u_np = u_encoder_inputs_from_row(row, stats_u, u_vocabs, u_cfg)
    xu = torch.from_numpy(x_u_np).to(device)
    catu = {k: torch.from_numpy(cat_u_np[k]).long().to(device) for k in sorted(u_vocabs.keys())}
    base_z = z_base_x_num_from_row(row, z_num_order, stats_z, u_cols_tpl)
    cat_z_np = z_categoricals_from_row(row, z_vocabs)

    S = int(n_samples)
    u_pack = sample_u_shared_gaussian_vm(
        u_model,
        xu,
        catu,
        n_samples=S,
        T_va=T_va,
        T_ang=T_ang_u,
        kappa_max=kappa_max_u,
        target_means=tm_u,
        target_stds=ts_u,
        generator=g_gen_u,
        d_tilde_deg_limits=d_tilde_deg_limits,
    )
    u_np = np.stack(
        [
            u_pack["v_ss_tilde"].cpu().numpy(),
            u_pack["a_tilde"].cpu().numpy(),
            u_pack["d_tilde"].cpu().numpy(),
        ],
        axis=1,
    )
    x_z_np = patch_z_x_num_with_u(base_z, z_num_order, u_cols_tpl, u_np, stats_z)
    x_z = torch.from_numpy(x_z_np).to(device)
    catz = {
        k: torch.full((S,), int(cat_z_np[k][0]), device=device, dtype=torch.long)
        for k in sorted(z_vocabs.keys())
    }
    if z_trunc_ex:
        z_pack = sample_z_hybrid_trunc_ex_batched(
            z_model,
            x_z,
            catz,
            n_samples_per_row=1,
            T_gauss=T_gauss_z,
            T_gauss_x=T_gx_f,
            T_gauss_y=T_gy_f,
            T_ang=T_ang_z,
            gauss_means=gm_z,
            gauss_stds=gs_z,
            generator=g_gen_z,
        )
        x_samp = z_pack["x"][:, 0].cpu().numpy()
        ey_samp = z_pack["e_y_star"][:, 0].cpu().numpy()
        psi_samp = z_pack["psi_deg"][:, 0].cpu().numpy()
        ex_samp = z_pack["e_x"][:, 0].cpu().numpy()
        th_samp = None
    else:
        z_pack = sample_z_hybrid_native_batched(
            z_model,
            x_z,
            catz,
            n_samples_per_row=1,
            T_gauss=T_gauss_z,
            T_gauss_x=T_gx_f,
            T_gauss_y=T_gy_f,
            T_ang=T_ang_z,
            gauss_means=gm_z,
            gauss_stds=gs_z,
            generator=g_gen_z,
        )
        x_samp = z_pack["x"][:, 0].cpu().numpy()
        ey_samp = z_pack["e_y_star"][:, 0].cpu().numpy()
        psi_samp = z_pack["psi_deg"][:, 0].cpu().numpy()
        th_samp = z_pack["theta_deg"][:, 0].cpu().numpy()
        ex_samp = None

    ev_list: list[float] = []
    la_list: list[float] = []
    sa_list: list[float] = []
    v_list: list[float] = []

    for j0 in range(0, S, decode_chunk):
        j1 = min(j0 + decode_chunk, S)
        inps: list[DecoderInputs] = []
        for j in range(j0, j1):
            uj = {
                "v_ss_tilde": float(u_np[j, 0]),
                "a_tilde": float(u_np[j, 1]),
                "d_tilde": float(u_np[j, 2]),
            }
            if ex_samp is not None:
                zj = {
                    "x": float(x_samp[j]),
                    "e_y_star": float(ey_samp[j]),
                    "psi_deg": float(psi_samp[j]),
                    "e_x": float(ex_samp[j]),
                }
            else:
                assert th_samp is not None
                zj = {
                    "x": float(x_samp[j]),
                    "e_y_star": float(ey_samp[j]),
                    "psi_deg": float(psi_samp[j]),
                    "theta_deg": float(th_samp[j]),
                }
            inps.append(
                DecoderInputs(
                    event_id=int(event_id),
                    batter_name=batter,
                    g=g_dec,
                    p=p_dec,
                    u=uj,
                    z=zj,
                )
            )
        try:
            df_batch = decode_bip_batch(inps)
        except DecoderInputError:
            continue
        ev_chunk: list[float] = []
        la_chunk: list[float] = []
        sa_chunk: list[float] = []
        for rel in range(len(inps)):
            out = df_batch.iloc[rel]
            if not bool(out.get("admissible", False)):
                continue
            ev = float(out["EV"])
            la = float(out["LA"])
            sa = float(out["SA"])
            if not (np.isfinite(ev) and np.isfinite(la) and np.isfinite(sa)):
                continue
            ev_chunk.append(ev)
            la_chunk.append(la)
            sa_chunk.append(sa)
        if not ev_chunk:
            continue
        ev_a = np.asarray(ev_chunk, dtype=np.float64)
        la_a = np.asarray(la_chunk, dtype=np.float64)
        sa_a = np.asarray(sa_chunk, dtype=np.float64)
        if enforce_sa_limits and d_tilde_deg_limits is not None:
            sa_ok = spray_angle_in_support(sa_a, limits=d_tilde_deg_limits)
            if not np.all(sa_ok):
                ev_a = ev_a[sa_ok]
                la_a = la_a[sa_ok]
                sa_a = sa_a[sa_ok]
            if ev_a.size == 0:
                continue
        if physics_calibrator is not None:
            cdf = _bootstrap_draws_df_for_calibration(row, ev_a, la_a, sa_a, physics_calibrator)
            if cdf.empty:
                continue
            out_c = physics_calibrator.transform_draws(cdf, group_col=physics_calibrator.group_col)
            ev_use = pd.to_numeric(out_c["EV_cal"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            ev_use = ev_a
        xw_arr = predict_xwobacon3d(ev_use, la_a, sa_a, model_bundle, calibrated=True)
        xw_arr = np.asarray(xw_arr, dtype=np.float64).ravel()
        ev_list.extend(ev_use.tolist())
        la_list.extend(la_a.tolist())
        sa_list.extend(sa_a.tolist())
        v_list.extend(xw_arr.tolist())

    return (
        np.asarray(ev_list, dtype=np.float64),
        np.asarray(la_list, dtype=np.float64),
        np.asarray(sa_list, dtype=np.float64),
        np.asarray(v_list, dtype=np.float64),
        S,
    )


def _save_hitter_distribution_png(
    ev: np.ndarray,
    la: np.ndarray,
    sa: np.ndarray,
    v: np.ndarray,
    path: Path,
    *,
    clip_sa_axis: bool = False,
) -> None:
    ev = np.asarray(ev, dtype=np.float64).ravel()
    la = np.asarray(la, dtype=np.float64).ravel()
    sa = np.asarray(sa, dtype=np.float64).ravel()
    xv = np.asarray(v, dtype=np.float64).ravel()
    if ev.size < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    bins = min(72, max(24, int(np.sqrt(float(ev.size)))))

    axes[0, 0].hist(ev, bins=bins, density=True, color="steelblue", alpha=0.85)
    axes[0, 0].set_title("EV (mph)")
    axes[0, 0].set_xlabel("EV")

    axes[0, 1].hist(la, bins=bins, density=True, color="darkorange", alpha=0.85)
    axes[0, 1].set_title("Launch angle (deg)")
    axes[0, 1].set_xlabel("LA")

    axes[1, 0].hist(sa, bins=bins, density=True, color="seagreen", alpha=0.85)
    axes[1, 0].set_title("Spray angle (deg)")
    axes[1, 0].set_xlabel("SA")
    if clip_sa_axis:
        axes[1, 0].set_xlim(SPRAY_ANGLE_DEG_LIMITS[0], SPRAY_ANGLE_DEG_LIMITS[1])

    axes[1, 1].hist(xv, bins=bins, density=True, color="purple", alpha=0.85)
    axes[1, 1].set_title("xwOBAcon (value)")
    axes[1, 1].set_xlabel("xwOBAcon")

    fig.suptitle(path.parent.name.replace("_", " ").title(), fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_ev_la_surface(ev: np.ndarray, la: np.ndarray, path_npz: Path, *, gridsize: int = 48) -> None:
    ev = np.asarray(ev, dtype=np.float64)
    la = np.asarray(la, dtype=np.float64)
    if len(ev) < 10:
        np.savez_compressed(path_npz, n=len(ev), hist=None, xedges=None, yedges=None)
        return
    h, xedges, yedges = np.histogram2d(ev, la, bins=gridsize)
    np.savez_compressed(path_npz, hist=h, xedges=xedges, yedges=yedges, n=len(ev))


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap SBI forward sim + LGBM value per hitter until convergence.")
    ap.add_argument(
        "--bip-parquet",
        type=Path,
        default=REPO_ROOT / "data/statcast_pybaseball/twelve_hitters_bip_full_statcast.parquet",
        help="Legacy Statcast pool (unused when --context-master is set).",
    )
    ap.add_argument(
        "--context-master",
        type=Path,
        default=None,
        help="bip_model-aligned context table (used with --bip-model-context-pool).",
    )
    ap.add_argument(
        "--bip-model-context-pool",
        action="store_true",
        help="Bootstrap + empirical from bip_model context pool with SA in [-45, 45] (non-default).",
    )
    ap.add_argument(
        "--enforce-spray-angle-limits",
        action="store_true",
        help="Reject d_tilde outside [-45, 45] at sample time and filter decoded SA.",
    )
    ap.add_argument(
        "--u-run-dir",
        type=Path,
        default=MMC2_ROOT / "sbi_forward_sim/outputs/p_u_given_g/20260407_193337Z",
    )
    ap.add_argument(
        "--z-run-dir",
        type=Path,
        default=MMC2_ROOT / "sbi_forward_sim/outputs/p_z_given_u_g/20260501_055549Z",
        help="Stage-z run dir (checkpoint.pt). Default: post-hoc–calibrated native_trunc_ex_vm.",
    )
    ap.add_argument(
        "--value-model-dir",
        type=Path,
        default=VALUE_MODEL_ROOT,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sbi_bootstrap_value_per_hitter_trunc_ex_vm",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=20260430)
    ap.add_argument("--n-samples-per-context", type=int, default=256, help="Joint u,z draws per bootstrap row.")
    ap.add_argument("--contexts-per-round", type=int, default=8, help="Bootstrap contexts per accumulation round.")
    ap.add_argument("--min-admissible", type=int, default=4000, help="Minimum admissible triples before testing convergence.")
    ap.add_argument("--max-admissible", type=int, default=80_000, help="Hard cap on stored admissible samples per player.")
    ap.add_argument("--rel-mean-tol", type=float, default=0.02, help="|mean_late-mean_early|/std(full) tolerance.")
    ap.add_argument("--ks-max", type=float, default=0.08, help="KS statistic threshold (first vs last 20%).")
    ap.add_argument("--converge-frac", type=float, default=0.15, help="Window fraction for mean comparison.")
    ap.add_argument("--decode-chunk-size", type=int, default=512)
    ap.add_argument("--players", type=str, default=None, help="Comma-separated batter_name substrings; default=all.")
    ap.add_argument(
        "--physics-calibrator-path",
        type=Path,
        default=None,
        help="Optional EVPhysicsCalibrator joblib; calibrates decoder EV before xwOBAcon.",
    )
    args = ap.parse_args()

    project_root = MMC2_ROOT / "sbi_forward_sim"
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    physics_calibrator: EVPhysicsCalibrator | None = None
    if args.physics_calibrator_path is not None:
        cp = Path(args.physics_calibrator_path).resolve()
        if not cp.is_file():
            raise FileNotFoundError(f"--physics-calibrator-path not found: {cp}")
        physics_calibrator = EVPhysicsCalibrator.load(cp)

    u_run = args.u_run_dir.resolve()
    z_run = args.z_run_dir.resolve()
    u_ckpt = _load_ckpt(u_run / "checkpoint.pt")
    z_ckpt = _load_ckpt(z_run / "checkpoint.pt")
    u_cfg = u_ckpt["config"]
    z_cfg = z_ckpt["config"]
    z_trunc_ex = str(z_ckpt.get("model_family", "")) == "hybrid_gauss_vonmises_trunc_ex"
    u_vocabs: dict = u_ckpt["vocabs"]
    z_vocabs: dict = z_ckpt["vocabs"]

    z_frozen_path = z_run / "feature_contract_frozen.json"
    if z_frozen_path.is_file():
        z_fc = json.loads(z_frozen_path.read_text(encoding="utf-8"))
    else:
        z_fc = load_feature_contract_z(z_cfg, project_root)
    z_num_order = list(z_fc["x_numeric_zscore_column_order"])
    u_cols_tpl = ("v_ss_tilde", "a_tilde", "d_tilde")

    master_path = (
        Path(args.context_master).resolve()
        if args.context_master is not None
        else (project_root / u_cfg["paths"]["context_master"]).resolve()
    )
    if not master_path.is_file():
        raise FileNotFoundError(master_path)
    master = pd.read_parquet(master_path)
    stats_u = _prepare_stats_u(project_root, u_cfg, master)
    stats_z = _prepare_stats_z(project_root, z_cfg, z_fc, master)

    train_u, _, _, _ = load_all_u_data(u_cfg, project_root, vocabs_override=u_vocabs)
    if z_trunc_ex:
        train_z, _, _, _ = load_all_z_data_native_trunc_ex(
            z_cfg, project_root, vocabs_override=z_vocabs, feature_contract=z_fc
        )
    else:
        train_z, _, _, _ = load_all_z_data_native_circular(
            z_cfg, project_root, vocabs_override=z_vocabs, feature_contract=z_fc
        )

    u_model = _build_u_model(u_cfg, u_vocabs, device)
    u_model.load_state_dict(u_ckpt["model_state"])
    u_model.eval()
    if z_trunc_ex:
        z_model = _build_z_model_trunc_ex(z_cfg, z_vocabs, device, feature_contract=z_fc, z_project_root=project_root)
    else:
        z_model = _build_z_model(z_cfg, z_vocabs, device, feature_contract=z_fc, z_project_root=project_root)
    z_model.load_state_dict(z_ckpt["model_state"])
    z_model.eval()

    T_va = float(u_ckpt["temperature_T_va"])
    T_ang_u = float(u_ckpt["temperature_T_ang"])
    kappa_max_u = float(u_ckpt.get("kappa_max", u_cfg["model"].get("kappa_max", 120.0)))
    T_gauss_z = float(z_ckpt["temperature_T_gauss"])
    T_ang_z = float(z_ckpt["temperature_T_ang"])
    T_gx = z_ckpt.get("temperature_T_gauss_x")
    T_gy = z_ckpt.get("temperature_T_gauss_y")
    T_gx_f = float(T_gx) if T_gx is not None else None
    T_gy_f = float(T_gy) if T_gy is not None else None

    tm_u = torch.tensor(train_u.target_means, dtype=torch.float32, device=device)
    ts_u = torch.tensor(train_u.target_stds, dtype=torch.float32, device=device)
    gm_z = torch.tensor(train_z.gauss_means, dtype=torch.float32, device=device)
    gs_z = torch.tensor(train_z.gauss_stds, dtype=torch.float32, device=device)

    model_bundle = load_batted_ball_value_model(args.value_model_dir.resolve())

    pc_path = project_root / "data_processed" / "sbi_player_constants.parquet"
    if not pc_path.is_file():
        raise FileNotFoundError(pc_path)
    pc = pd.read_parquet(pc_path)
    pc_by = pc.set_index("batter_name", drop=False)

    bip_path = args.bip_parquet.resolve()
    if not bip_path.is_file():
        raise FileNotFoundError(bip_path)

    use_bip_model_pool = bool(args.bip_model_context_pool)
    enforce_sa = bool(args.enforce_spray_angle_limits)
    d_limits = SPRAY_ANGLE_DEG_LIMITS if enforce_sa else None

    bip = pd.read_parquet(bip_path)
    if "batter_name" not in bip.columns:
        raise ValueError("BIP parquet must contain batter_name")
    bip["batter_name"] = bip["batter_name"].astype(str).str.strip()

    if use_bip_model_pool:
        context_pool = _build_bootstrap_context_pool(master, players=None)
        players_all = sorted(context_pool["batter_name"].unique())
    else:
        context_pool = None
        players_all = sorted(bip["batter_name"].unique())

    if args.players:
        subs = [s.strip() for s in args.players.split(",") if s.strip()]
        players = [p for p in players_all if any(sub.lower() in p.lower() for sub in subs)]
    else:
        players = players_all

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "bip_parquet": str(bip_path),
        "context_master": str(master_path),
        "bootstrap_pool": "bip_model_context" if use_bip_model_pool else "statcast_bip_parquet",
        "context_pool_rows": int(len(context_pool)) if context_pool is not None else int(len(bip)),
        "spray_angle_deg_limits": list(d_limits) if d_limits else None,
        "empirical_filter": (
            "bip_model_training_event_mask + spray_angle_deg in [-45, 45]"
            if use_bip_model_pool
            else "statcast_bip_parquet launch_speed/launch_angle + hc_x/hc_y spray"
        ),
        "u_run_dir": str(u_run),
        "z_run_dir": str(z_run),
        "z_model_family": str(z_ckpt.get("model_family", "")),
        "z_trunc_ex": z_trunc_ex,
        "value_model_dir": str(args.value_model_dir.resolve()),
        "seed": args.seed,
        "n_samples_per_context": args.n_samples_per_context,
        "contexts_per_round": args.contexts_per_round,
        "min_admissible": args.min_admissible,
        "max_admissible": args.max_admissible,
        "physics_calibrator_path": str(Path(args.physics_calibrator_path).resolve()) if args.physics_calibrator_path else None,
    }
    (out_root / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    synth_base = 9_000_000_000
    decode_chunk = max(1, int(args.decode_chunk_size))

    for pi, player in enumerate(players):
        pdir = out_root / _safe_dir_name(player)
        pdir.mkdir(parents=True, exist_ok=True)
        if use_bip_model_pool:
            assert context_pool is not None
            pool = context_pool.loc[context_pool["batter_name"].eq(player)].reset_index(drop=True)
        else:
            pool = bip.loc[bip["batter_name"].eq(player)].reset_index(drop=True)
        n_pool = len(pool)
        if n_pool < 5:
            print(f"skip {player!r}: pool too small ({n_pool})")
            continue
        if player not in pc_by.index:
            print(f"skip {player!r}: missing sbi_player_constants")
            continue
        prow = pc_by.loc[player]
        if isinstance(prow, pd.DataFrame):
            prow = prow.iloc[0]
        # Stage-z / u encoders expect full player-constant block including I0 (decoder ignores I0 per contract).
        p_const = {c: float(prow[c]) for c in pc.columns if c != "batter_name"}

        obs_ev = pd.to_numeric(pool["launch_speed"], errors="coerce")
        obs_la = pd.to_numeric(pool["launch_angle"], errors="coerce")
        if use_bip_model_pool:
            if "spray_angle_deg" in pool.columns:
                obs_sa = pd.to_numeric(pool["spray_angle_deg"], errors="coerce")
            else:
                obs_sa = pd.to_numeric(pool["spray_angle_obs_deg"], errors="coerce")
            m_obs = obs_ev.notna() & obs_la.notna() & obs_sa.notna()
            if enforce_sa:
                m_obs = m_obs & spray_angle_in_support(obs_sa, limits=SPRAY_ANGLE_DEG_LIMITS)
        else:
            obs_sa = _spray_angle_deg_series(pool["hc_x"], pool["hc_y"])
            m_obs = obs_ev.notna() & obs_la.notna() & obs_sa.notna()
        if m_obs.sum() > 0:
            v_obs = predict_xwobacon3d(
                obs_ev[m_obs].to_numpy(),
                obs_la[m_obs].to_numpy(),
                obs_sa[m_obs].to_numpy(),
                model_bundle,
                calibrated=True,
            )
            v_obs = np.asarray(v_obs, dtype=np.float64).ravel()
            np.savez_compressed(
                pdir / "baseline_observed_bip.npz",
                launch_speed=obs_ev[m_obs].to_numpy(),
                launch_angle=obs_la[m_obs].to_numpy(),
                spray_angle_deg=obs_sa[m_obs].to_numpy(),
                xwobacon=v_obs,
            )

        ev_acc: list[float] = []
        la_acc: list[float] = []
        sa_acc: list[float] = []
        v_acc: list[float] = []
        rounds: list[dict[str, Any]] = []
        converged = False
        g_gen_u = torch.Generator(device=device)
        g_gen_u.manual_seed(args.seed + pi * 100_003)
        g_gen_z = torch.Generator(device=device)
        g_gen_z.manual_seed(args.seed + pi * 100_003 + 17)

        round_id = 0
        max_rounds_no_admissible = 2500
        while len(v_acc) < int(args.max_admissible):
            round_id += 1
            if len(v_acc) == 0 and round_id > max_rounds_no_admissible:
                print(
                    f"  abort {player!r}: no admissible draws after {max_rounds_no_admissible} rounds "
                    "(check decoder inputs / sampling)."
                )
                break
            idxs = rng.integers(0, n_pool, size=int(args.contexts_per_round))
            n_adm_round = 0
            n_att_round = 0
            for local_i, ri in enumerate(idxs):
                stat = pool.iloc[int(ri)]
                if "event_id" in stat.index and pd.notna(stat.get("event_id")):
                    eid = int(stat["event_id"])
                else:
                    eid = synth_base + pi * 50_000_000 + round_id * 10_000 + local_i
                try:
                    row = statcast_row_to_model_row(stat, event_id=eid, p_const=p_const)  # type: ignore[arg-type]
                except (ValueError, KeyError) as ex:
                    print(f"  row build fail {player}: {ex}")
                    continue
                batter = str(row["batter_name"])
                try:
                    ev_a, la_a, sa_a, v_a, attempted = sample_decode_value_batch(
                        row,
                        event_id=eid,
                        batter=batter,
                        n_samples=int(args.n_samples_per_context),
                        u_model=u_model,
                        z_model=z_model,
                        u_cfg=u_cfg,
                        stats_u=stats_u,
                        u_vocabs=u_vocabs,
                        stats_z=stats_z,
                        z_vocabs=z_vocabs,
                        z_num_order=z_num_order,
                        u_cols_tpl=u_cols_tpl,
                        device=device,
                        T_va=T_va,
                        T_ang_u=T_ang_u,
                        kappa_max_u=kappa_max_u,
                        T_gauss_z=T_gauss_z,
                        T_ang_z=T_ang_z,
                        T_gx_f=T_gx_f,
                        T_gy_f=T_gy_f,
                        tm_u=tm_u,
                        ts_u=ts_u,
                        gm_z=gm_z,
                        gs_z=gs_z,
                        g_gen_u=g_gen_u,
                        g_gen_z=g_gen_z,
                        model_bundle=model_bundle,
                        decode_chunk=decode_chunk,
                        z_trunc_ex=z_trunc_ex,
                        physics_calibrator=physics_calibrator,
                        d_tilde_deg_limits=d_limits,
                        enforce_sa_limits=enforce_sa,
                    )
                except Exception as ex:
                    print(f"  sample/decode fail {player} eid={eid}: {ex}")
                    continue
                n_att_round += attempted
                n_adm_round += len(v_a)
                ev_acc.extend(ev_a.tolist())
                la_acc.extend(la_a.tolist())
                sa_acc.extend(sa_a.tolist())
                v_acc.extend(v_a.tolist())
                if len(v_acc) >= int(args.max_admissible):
                    ev_acc = ev_acc[: int(args.max_admissible)]
                    la_acc = la_acc[: int(args.max_admissible)]
                    sa_acc = sa_acc[: int(args.max_admissible)]
                    v_acc = v_acc[: int(args.max_admissible)]
                    break

            v_arr = np.asarray(v_acc, dtype=np.float64)
            ok_conv = convergence_ok(
                v_arr,
                min_n=int(args.min_admissible),
                rel_mean_tol=float(args.rel_mean_tol),
                ks_max=float(args.ks_max),
                frac=float(args.converge_frac),
            )
            rounds.append(
                {
                    "round": round_id,
                    "n_admissible_total": int(len(v_acc)),
                    "n_admissible_round": int(n_adm_round),
                    "n_attempted_joint_round": int(n_att_round),
                    "mean_xwobacon": float(np.mean(v_arr)) if len(v_arr) else float("nan"),
                    "converged": bool(ok_conv and len(v_acc) >= int(args.min_admissible)),
                }
            )
            print(
                f"{player} round {round_id}: adm_total={len(v_acc)} "
                f"adm_round={n_adm_round} conv={ok_conv and len(v_acc) >= int(args.min_admissible)}"
            )
            if ok_conv and len(v_acc) >= int(args.min_admissible):
                converged = True
                break

        ev_f = np.asarray(ev_acc, dtype=np.float64)
        la_f = np.asarray(la_acc, dtype=np.float64)
        sa_f = np.asarray(sa_acc, dtype=np.float64)
        v_f = np.asarray(v_acc, dtype=np.float64)
        pd.DataFrame({"EV": ev_f, "LA": la_f, "SA": sa_f, "xwobacon": v_f}).to_parquet(
            pdir / "simulated_ev_la_sa_value.parquet", index=False
        )
        np.savez_compressed(
            pdir / "value_function_distribution.npz",
            xwobacon=v_f,
            converged=converged,
            n_rounds=round_id,
        )
        save_ev_la_surface(ev_f, la_f, pdir / "surface_ev_la_hist2d.npz")
        if len(sa_f) > 5:
            sh, sedges = np.histogram(sa_f, bins=64)
            np.savez_compressed(pdir / "surface_sa_marginal.npz", hist=sh, edges=sedges)
        else:
            np.savez_compressed(pdir / "surface_sa_marginal.npz", hist=np.array([]), edges=np.array([]))
        (pdir / "convergence_rounds.json").write_text(json.dumps(rounds, indent=2), encoding="utf-8")
        _save_hitter_distribution_png(
            ev_f,
            la_f,
            sa_f,
            v_f,
            pdir / "distributions_ev_la_sa_xwobacon.png",
            clip_sa_axis=enforce_sa,
        )

    print("Done. Outputs under", out_root)


if __name__ == "__main__":
    main()
