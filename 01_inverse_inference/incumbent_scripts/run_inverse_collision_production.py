#!/usr/bin/env python3
"""Production runner for exact-root inverse-collision MCMC.

This script ports the patched notebook workflow into a resumable, per-event
production pipeline while preserving the exact-root model architecture.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import truncnorm


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
PSI_SUPPORT_DEG = (-70.0, 70.0)
EY_BOUNDS = (0.15, 0.60)
EY_SHIFT = 0.05
EY_SD = 0.04
ROOT_TOL = 1e-10
DENOM_TOL = 1e-9

MIN_PLAYER_CAL_ROWS = 120
SHRINK_KAPPA = 300.0
MAX_CAL_ROWS = 120000


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description="Run inverse-collision production MCMC")

    # Core run args
    p.add_argument("--input-data-path", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--event-ids", default="")
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--random-seed", type=int, default=20260408)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--n-iter", type=int, default=700)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--n-chains", type=int, default=1)
    p.add_argument("--max-init-attempts", type=int, default=1200)
    p.add_argument("--retry-failed-init", action="store_true")
    p.add_argument("--max-init-attempts-retry", type=int, default=3000)
    p.add_argument("--init-retry-rounds", type=int, default=1)

    # Proposal tuning args
    p.add_argument("--warmup-iters", type=int, default=120)
    p.add_argument("--tune-proposals", action="store_true")
    p.add_argument("--target-accept-low", type=float, default=0.10)
    p.add_argument("--target-accept-high", type=float, default=0.25)
    p.add_argument("--proposal-sd-x", type=float, default=0.9)
    p.add_argument("--proposal-sd-measurement-block", type=float, default=1.0)
    p.add_argument("--proposal-sd-e-y", type=float, default=1.0)
    p.add_argument("--proposal-sd-phi", type=float, default=1.0)
    p.add_argument("--proposal-sd-attack-angle", type=float, default=1.0)
    p.add_argument("--proposal-sd-bat-speed", type=float, default=1.0)

    # Output control args
    p.add_argument("--save-full-proposal-trace", action="store_true")
    p.add_argument("--proposal-trace-stride", type=int, default=1)
    p.add_argument("--save-per-event-parquet", action="store_true")
    p.add_argument("--save-combined-parquet", action="store_true")
    p.add_argument("--save-csv-summaries", action="store_true")
    p.add_argument("--compression-codec", default="zstd")
    p.add_argument("--make-train-exports", action="store_true")
    p.add_argument("--make-calibration-exports", action="store_true")
    p.add_argument("--make-sbc-hooks", action="store_true")

    # Split / export args
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=20260409)
    p.add_argument("--group-splits-by", choices=["event", "hitter"], default="event")
    p.add_argument("--strict-screening", action="store_true")
    p.add_argument("--basic-screening", action="store_true")

    # Calibration args
    p.add_argument("--run-posterior-predictive-checks", action="store_true")
    p.add_argument("--run-sbc-if-simulator-available", action="store_true")
    p.add_argument("--n-sbc-sims", type=int, default=100)
    p.add_argument("--n-ppc-draws", type=int, default=250)

    # Internals
    p.add_argument(
        "--source-script-path",
        default="/home/evangoforth03/Bayesian Research/MCMC 2/run_mcmc_posterior_bank.py",
    )
    return p


# ---------------------------------------------------------------------
# Logging / I/O helpers
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    """Print timestamp-less production log line."""
    print(msg, flush=True)


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if not np.isfinite(seconds) or seconds < 0:
        return "??:??:??"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def progress_bar(done: int, total: int, width: int = 28) -> str:
    """Return fixed-width ascii progress bar."""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    frac = min(1.0, max(0.0, done / total))
    fill = int(round(width * frac))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_table(df: pd.DataFrame, path: Path, compression: str = "zstd") -> None:
    """Write dataframe to parquet, fallback to csv on engine errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False, compression=compression)
            return
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table extension: {path}")


def clean_export_df(
    df: pd.DataFrame,
    preferred_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Standardize export schema and deterministic ordering."""
    out = df.copy()
    out = out[[c for c in out.columns if not str(c).startswith("Unnamed:")]]
    preferred_cols = preferred_cols or []
    categorical_cols = categorical_cols or []

    for c in categorical_cols:
        if c in out.columns:
            out[c] = out[c].astype("object").where(out[c].notna(), "missing")
            if c == "failure_reason":
                out[c] = out[c].replace("", "accepted")
            else:
                out[c] = out[c].replace("", "missing")

    if "event_id" in out.columns:
        out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").astype("Int64")
    if "branch_index" in out.columns:
        out["branch_index"] = pd.to_numeric(out["branch_index"], errors="coerce").astype("Int64")

    cols = [c for c in preferred_cols if c in out.columns]
    rest = sorted([c for c in out.columns if c not in cols])
    return out[cols + rest]


# ---------------------------------------------------------------------
# Math/calibration helpers
# ---------------------------------------------------------------------
def wrap180(deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-180, 180)."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def _fit_spline(x, y, smooth=None, min_points=16, k=3):
    """Fit robust binned spline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < min_points:
        return None

    nbins = max(25, min(140, int(np.sqrt(len(x)))))
    edges = np.linspace(float(np.min(x)), float(np.max(x)), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins)
    sx = np.bincount(idx, weights=x, minlength=nbins)
    sy = np.bincount(idx, weights=y, minlength=nbins)
    keep = cnt > 0
    xb = sx[keep] / cnt[keep]
    yb = sy[keep] / cnt[keep]
    if len(xb) < min_points:
        return None

    order = np.argsort(xb)
    xb, yb = xb[order], yb[order]
    xb_u, inv = np.unique(xb, return_inverse=True)
    yb_u = np.bincount(inv, weights=yb) / np.maximum(1, np.bincount(inv))
    if len(xb_u) < min_points:
        return None
    if smooth is None:
        smooth = len(xb_u) * np.var(yb_u) * 0.06
    return UnivariateSpline(xb_u, yb_u, s=smooth, k=min(k, len(xb_u) - 1), ext="const")


@dataclass
class CalibModel:
    """Per-player blended calibration model."""

    player_name: str
    n_rows: int
    weight_player: float
    g_aa_pooled: Any
    g_bs_pooled: Any
    f_aa_pooled: Any
    f_bs_pooled: Any
    g_aa_player: Any
    g_bs_player: Any
    f_aa_player: Any
    f_bs_player: Any

    def _blend(self, pooled, player, x):
        if player is None:
            return pooled(x)
        return self.weight_player * player(x) + (1.0 - self.weight_player) * pooled(x)

    def correction(self, spray_deg, delta_deg):
        """Return deterministic shifts for AA and BS."""
        f_aa_delta = self._blend(self.f_aa_pooled, self.f_aa_player, delta_deg)
        f_bs_delta = self._blend(self.f_bs_pooled, self.f_bs_player, delta_deg)
        f_aa_zero = float(self._blend(self.f_aa_pooled, self.f_aa_player, np.array([0.0]))[0])
        f_bs_zero = float(self._blend(self.f_bs_pooled, self.f_bs_player, np.array([0.0]))[0])
        return f_aa_delta - f_aa_zero, f_bs_delta - f_bs_zero


def _fit_single_calibrator(df_sub: pd.DataFrame):
    """Fit spray trend and delta residual splines."""
    g_aa = _fit_spline(df_sub["spray_angle_deg"], df_sub["attack_angle"])
    g_bs = _fit_spline(df_sub["spray_angle_deg"], df_sub["bat_speed"])
    if g_aa is None or g_bs is None:
        return None
    aa_res = df_sub["attack_angle"].to_numpy() - g_aa(df_sub["spray_angle_deg"].to_numpy())
    bs_res = df_sub["bat_speed"].to_numpy() - g_bs(df_sub["spray_angle_deg"].to_numpy())
    delta = wrap180(df_sub["attack_direction"].to_numpy() - df_sub["spray_angle_deg"].to_numpy())
    f_aa = _fit_spline(delta, aa_res)
    f_bs = _fit_spline(delta, bs_res)
    if f_aa is None or f_bs is None:
        return None
    return g_aa, g_bs, f_aa, f_bs


def build_calibration_models(bip: pd.DataFrame):
    """Build pooled + player models with EB shrinkage."""
    need = ["batter_name", "spray_angle_deg", "attack_direction", "attack_angle", "bat_speed"]
    cal = bip.dropna(subset=need).copy()
    cal = cal[(cal["spray_angle_deg"] >= -45.0) & (cal["spray_angle_deg"] <= 45.0)]
    if len(cal) > MAX_CAL_ROWS:
        cal = cal.sample(n=MAX_CAL_ROWS, random_state=42)

    pooled = _fit_single_calibrator(cal)
    if pooled is None:
        raise RuntimeError("Cannot fit pooled calibration")
    g_aa_pool, g_bs_pool, f_aa_pool, f_bs_pool = pooled

    models = {}
    diag = []
    for bn, grp in cal.groupby("batter_name", sort=False):
        n = len(grp)
        w = float(n / (n + SHRINK_KAPPA))
        pf = _fit_single_calibrator(grp) if n >= MIN_PLAYER_CAL_ROWS else None
        if pf is None:
            g_aa_p = g_bs_p = f_aa_p = f_bs_p = None
            w_use = 0.0
        else:
            g_aa_p, g_bs_p, f_aa_p, f_bs_p = pf
            w_use = w
        models[bn] = CalibModel(
            player_name=bn,
            n_rows=n,
            weight_player=w_use,
            g_aa_pooled=g_aa_pool,
            g_bs_pooled=g_bs_pool,
            f_aa_pooled=f_aa_pool,
            f_bs_pooled=f_bs_pool,
            g_aa_player=g_aa_p,
            g_bs_player=g_bs_p,
            f_aa_player=f_aa_p,
            f_bs_player=f_bs_p,
        )
        diag.append(
            {
                "batter_name": bn,
                "calibration_sample_size": n,
                "shrinkage_weight": w_use,
            }
        )
    return models, pd.DataFrame(diag)


def _safe_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation with safeguards."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    a, b = x[:-1], x[1:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _ess_from_rho1(n: int, rho1: float) -> float:
    """Approximate ESS from lag-1 rho."""
    if n <= 1 or not np.isfinite(rho1):
        return np.nan
    den = (1.0 + rho1)
    if abs(den) < 1e-12:
        return np.nan
    val = n * (1.0 - rho1) / den
    return float(max(1.0, min(float(n), val))) if np.isfinite(val) else np.nan


# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------
def load_and_filter_events(data_path: Path, bank_module):
    """Load raw data and apply notebook-equivalent eligibility filters."""
    base = pd.read_pickle(data_path)
    if "event_id" not in base.columns:
        base = base.reset_index(drop=True)
        base["event_id"] = np.arange(len(base), dtype=int)

    pack = bank_module.build_hitters_and_bip_model(base.copy(), verbose=False)
    hitters_all, _meta, _bip_sub, bip, _ord = pack

    if "stand" in bip.columns:
        bip = bip[bip["stand"].astype(str).str.upper().str.startswith("R")]
    if "batter_name" in bip.columns:
        nm = bip["batter_name"].astype(str).str.strip().str.lower()
        bip = bip[nm != "ozzie albies"]
    if "spray_angle_deg" in bip.columns:
        bip = bip[(bip["spray_angle_deg"] >= -45.0) & (bip["spray_angle_deg"] <= 45.0)]

    bip = bip.reset_index(drop=True)
    if "event_id" not in bip.columns:
        bip["event_id"] = np.arange(len(bip), dtype=int)

    keep_names = set(bip["batter_name"].astype(str).unique()) if "batter_name" in bip.columns else set()
    hitters_by_name = {h.name: h for h in hitters_all if h.name in keep_names}
    return base, bip, hitters_by_name


def parse_event_ids(arg: str) -> set[int]:
    """Parse event ids arg as CSV list or file path."""
    if not arg:
        return set()
    p = Path(arg)
    if p.exists():
        txt = p.read_text().strip().splitlines()
        ids = set()
        for line in txt:
            for tok in line.replace(",", " ").split():
                if tok.strip():
                    ids.add(int(tok))
        return ids
    return {int(x.strip()) for x in arg.split(",") if x.strip()}


def select_events(
    bip: pd.DataFrame,
    event_ids: set[int],
    max_events: int,
    seed: int,
) -> pd.DataFrame:
    """Select events by explicit ids or random subsample."""
    out = bip.copy()
    if event_ids:
        out = out[out["event_id"].isin(event_ids)].copy()
    if max_events and len(out) > max_events:
        out = out.sample(n=max_events, random_state=seed).copy()
    return out.sort_values("event_id").reset_index(drop=True)


# ---------------------------------------------------------------------
# Model core (patched exact-root path)
# ---------------------------------------------------------------------
def sample_e_y_star(x_in: float, hitter, bank_module, rng: np.random.Generator):
    """Sample e_y in proposal state."""
    mu = float(bank_module.e_y_player(x_in, hitter) + EY_SHIFT)
    a = (EY_BOUNDS[0] - mu) / EY_SD
    b = (EY_BOUNDS[1] - mu) / EY_SD
    val = float(truncnorm.rvs(a, b, loc=mu, scale=EY_SD, random_state=rng))
    logp = float(truncnorm.logpdf(val, a, b, loc=mu, scale=EY_SD))
    return val, logp


def build_transformed_measurements(ev_star: dict, calib_model: CalibModel, bank_module):
    """Build deterministic transformed kinematics from sampled measurement block."""
    phi_rad, _hcx, _hcy = bank_module.spray_phi_from_hc(ev_star["hc_x"], ev_star["hc_y"])
    phi_star = float(np.degrees(phi_rad))
    attack_dir_obs_star = float(ev_star.get("attack_direction_deg", np.nan))
    attack_angle_obs_star = float(ev_star.get("attack_angle_deg", np.nan))
    bat_speed_obs_star = float(ev_star.get("bat_speed_mph", np.nan))

    delta_star = float(wrap180(attack_dir_obs_star - phi_star))
    aa_shift, bs_shift = calib_model.correction(np.array([phi_star]), np.array([delta_star]))
    aa_shift = float(np.asarray(aa_shift)[0])
    bs_shift = float(np.asarray(bs_shift)[0])

    d_tilde = phi_star
    a_tilde = attack_angle_obs_star - aa_shift
    v_ss_tilde = bat_speed_obs_star - bs_shift

    pitch = bank_module.build_incoming_pitch_from_sensor(
        vx0=ev_star["vx0"],
        vy0=ev_star["vy0"],
        vz0=ev_star["vz0"],
        ax=ev_star["ax"],
        ay=ev_star["ay"],
        az=ev_star["az"],
        release_pos_y=ev_star["release_pos_y"],
    )
    omega_minus = bank_module.relevant_omega_minus_from_sensor(
        release_spin_rate_rpm=ev_star["release_spin_rate_rpm"],
        spin_axis_deg=ev_star["spin_axis_deg"],
        phi_rad=phi_rad,
        t_contact=pitch["t_contact"],
        asr=ev_star.get("active_spin_ratio", bank_module.ASR_DEFAULT),
    )

    return {
        "phi_star_deg": phi_star,
        "attack_direction_obs_star_deg": attack_dir_obs_star,
        "attack_angle_obs_star_deg": attack_angle_obs_star,
        "bat_speed_obs_star_mph": bat_speed_obs_star,
        "delta_star_deg": delta_star,
        "attack_direction_calib_deg": d_tilde,
        "attack_angle_calib_to_spray_deg": float(a_tilde),
        "bat_speed_calib_to_spray_mph": float(v_ss_tilde),
        "theta_obs_deg": float(ev_star["launch_angle_deg"]),
        "s_obs_mph": float(ev_star["launch_speed_mph"]),
        "omega_minus": float(omega_minus),
        "vB_x_fps": float(pitch["vin_x"]),
        "vB_y_fps": float(pitch["vin_y"]),
        "vB_z_fps": float(pitch["vin_z"]),
    }


def bat_vector_from_x(x_in: float, trans: dict, hitter, bank_module):
    """Build transformed bat vector from x and transformed state."""
    L = float(hitter.bat_length)
    scale = (x_in - 6.0) / (L - 12.0)
    v_coll_mph = float(trans["bat_speed_calib_to_spray_mph"] * scale)
    phi = math.radians(float(trans["phi_star_deg"]))
    a = math.radians(float(trans["attack_angle_calib_to_spray_deg"]))
    vx = v_coll_mph * math.cos(a) * math.sin(phi)
    vy = v_coll_mph * math.cos(a) * math.cos(phi)
    vz = v_coll_mph * math.sin(a)
    return {
        "v_coll_mph": v_coll_mph,
        "v_coll_fps": float(v_coll_mph * bank_module.MPH_TO_FPS),
        "vx_bat_mph": float(vx),
        "vy_bat_mph": float(vy),
        "vz_bat_mph": float(vz),
    }


def classify_failure_stage(row: dict[str, Any]) -> dict[str, bool]:
    """Stage-based reject classification."""
    fr = str(row.get("failure_reason", "accepted") or "accepted")
    accepted = fr == "accepted"

    stage1 = fr in {"no_analytic_root_amp_below_tol", "no_exact_root", "amp_below_root_tol"}
    stage2 = fr in {"root_outside_psi_support"}
    stage3 = fr in {
        "vt_over_vn_ge_1",
        "e_x_out_of_bounds",
        "regime_gross_slip",
        "regime_violation",
        "nonpositive_Vn_plus",
        "nonpositive_delta_n",
        "geometry_D_out_of_bounds",
        "branch_sign_failure",
    }
    stage4 = fr in {"bad_denom_Ft", "nonfinite_log_prior_ex", "nonfinite_log_target"}
    hard_before_ft = stage1 or stage2 or fr in {"nonpositive_Vn_plus", "vt_over_vn_ge_1", "nonpositive_delta_n"}
    hard_before_log = stage1 or stage2 or stage3 or fr == "bad_denom_Ft"

    true_num = (
        fr
        in {
            "nan_in_trig",
            "nan_in_root",
            "nan_in_Ft",
            "nan_in_Fw",
            "near_zero_Ft_denom",
            "near_zero_Fw_denom",
            "nonfinite_log_prior_x",
            "nonfinite_log_prior_measurement",
            "nonfinite_log_prior_ey",
            "nonfinite_log_prior_ex",
            "nonfinite_log_target",
        }
        and (not hard_before_log)
    )

    if accepted:
        stage1 = stage2 = stage3 = stage4 = False
        hard_before_ft = hard_before_log = False
        true_num = False

    return {
        "failed_stage_1_root": bool(stage1),
        "failed_stage_2_support": bool(stage2),
        "failed_stage_3_admissibility": bool(stage3),
        "failed_stage_4_downstream": bool(stage4),
        "true_numerical_failure": bool(true_num),
        "hard_reject_before_Ft": bool(hard_before_ft),
        "hard_reject_before_log_target": bool(hard_before_log),
    }


def solve_exact_root_state(
    ev_obs: dict,
    ev_star: dict,
    trans: dict,
    hitter,
    x_in: float,
    e_y_star: float,
    logp_measure: float,
    logp_ey: float,
    bank_module,
) -> dict[str, Any]:
    """Patched exact-root solver (kept consistent with notebook)."""
    out: dict[str, Any] = {
        "x": float(x_in),
        "e_y_star": float(e_y_star),
        "log_prior_measurement": float(logp_measure),
        "log_prior_e_y": float(logp_ey),
        "log_prior_x": float(-math.log(11.0)),
        "log_prior_ex": np.nan,
        "log_target": np.nan,
        "root_exists": 0,
        "admissible": 0,
        "accepted_physics": 0,
        "failure_reason": "unknown",
        "branch_idx": -1,
        "psi_rel_definition": "legacy_bat_only_from_vb_components",
    }

    theta = math.radians(float(trans["theta_obs_deg"]))
    s_obs = float(trans["s_obs_mph"]) * bank_module.MPH_TO_FPS
    phi = math.radians(float(trans["phi_star_deg"]))

    vB_x, vB_y, vB_z = float(trans["vB_x_fps"]), float(trans["vB_y_fps"]), float(trans["vB_z_fps"])
    vB_r = vB_x * math.sin(phi) + vB_y * math.cos(phi)
    vB_t = vB_x * math.cos(phi) - vB_y * math.sin(phi)

    bvec = bat_vector_from_x(x_in, trans, hitter, bank_module)
    vb_x = bvec["vx_bat_mph"] * bank_module.MPH_TO_FPS
    vb_y = bvec["vy_bat_mph"] * bank_module.MPH_TO_FPS
    vb_z = bvec["vz_bat_mph"] * bank_module.MPH_TO_FPS
    vb_r = vb_x * math.sin(phi) + vb_y * math.cos(phi)
    vb_t = vb_x * math.cos(phi) - vb_y * math.sin(phi)

    dv_r = vb_r - vB_r
    dv_z = vb_z - vB_z

    R_loc = bank_module.R_player(x_in, hitter)
    r_y_val = float(
        bank_module.r_y_of_x(x_in, hitter.x_cm, hitter.m_ball_oz, hitter.bat_weight_oz, hitter.I0_oz_in2)
    )
    r_x_val = float(
        bank_module.r_x_of_x(
            x_in,
            hitter.x_cm,
            hitter.m_ball_oz,
            hitter.alpha,
            hitter.bat_weight_oz,
            hitter.I0_oz_in2,
            R_loc,
            hitter.Iz_oz_in2,
        )
    )
    lam = (1.0 + e_y_star) / (1.0 + r_y_val)
    r_ball_ft = float(hitter.r_ball_in) / 12.0

    A = vB_r + lam * dv_r - s_obs * math.cos(theta)
    B = vB_z + lam * dv_z - s_obs * math.sin(theta)
    amp = math.hypot(A, B)

    out.update(
        {
            **bvec,
            "vB_r": float(vB_r),
            "vB_t": float(vB_t),
            "vb_r": float(vb_r),
            "vb_t": float(vb_t),
            "Delta_n": float(dv_r),
            "Delta_t": float(dv_z),
            "A": float(A),
            "B": float(B),
            "lambda_x": float(lam),
            "r_y_x": float(r_y_val),
            "r_x_x": float(r_x_val),
        }
    )

    if amp < ROOT_TOL:
        out["failure_reason"] = "no_analytic_root_amp_below_tol"
        out.update(classify_failure_stage(out))
        return out

    psi0 = math.atan2(-A, B)
    roots = [psi0, psi0 + math.pi, psi0 - math.pi]
    valid_roots = []
    for i, root in enumerate(roots):
        psi = ((root + math.pi) % (2.0 * math.pi)) - math.pi
        psi_deg = math.degrees(psi)
        if PSI_SUPPORT_DEG[0] <= psi_deg <= PSI_SUPPORT_DEG[1]:
            valid_roots.append((i, psi, psi_deg))

    if not valid_roots:
        out["failure_reason"] = "root_outside_psi_support"
        out.update(classify_failure_stage(out))
        return out

    out["root_exists"] = 1
    fail_counter: dict[str, int] = {}

    def bump(key: str) -> None:
        fail_counter[key] = fail_counter.get(key, 0) + 1

    for branch_idx, psi, psi_deg in valid_roots:
        cp, sp = math.cos(psi), math.sin(psi)
        vB_n = vB_r * cp + vB_z * sp
        vB_tn = -vB_r * sp + vB_z * cp
        vb_n = vb_r * cp + vb_z * sp
        vb_tn = -vb_r * sp + vb_z * cp
        d_n = dv_r * cp + dv_z * sp
        d_t = -dv_r * sp + dv_z * cp
        Vn_plus = s_obs * (math.cos(theta) * cp + math.sin(theta) * sp)
        Vt_plus = s_obs * (math.sin(theta) * cp - math.cos(theta) * sp)
        vt_over_vn_abs = np.inf if abs(Vn_plus) < DENOM_TOL else abs(Vt_plus) / abs(Vn_plus)

        psi_rel_legacy = math.atan2(vb_tn, vb_n)
        vrel_n_minus = vB_n - vb_n
        vrel_t_minus = vB_tn - vb_tn - r_ball_ft * trans["omega_minus"]
        psi_rel_relkin = math.atan2(vrel_t_minus, vrel_n_minus)
        D_in = bank_module.D_of_psi_rel(R_loc, hitter.r_ball_in, psi_rel_legacy)
        denom_ft = hitter.alpha * (d_t - r_ball_ft * trans["omega_minus"])

        e_x_raw = np.nan
        omega_plus = np.nan
        if abs(denom_ft) >= DENOM_TOL:
            e_x_raw = ((Vt_plus - vB_tn) * (1.0 + r_x_val) * (1.0 + hitter.alpha)) / denom_ft - 1.0
            omega_plus = trans["omega_minus"] + (
                ((vB_tn - Vt_plus) - (D_in / hitter.r_ball_in) * Vn_plus) / (hitter.alpha * r_ball_ft)
            )

        regime_info = bank_module.classify_regime_from_psi_rel(
            abs(psi_rel_legacy), bank_module.DEFAULT_MU, bank_module.DEFAULT_G2_BASEBALL, e_y_star
        )
        regime_label = regime_info["regime_label"]
        log_prior_ex = bank_module.log_prior_ex_from_regime(e_x_raw, regime_label)

        branch_diag = {
            "branch_idx": int(branch_idx),
            "psi_root_deg": float(psi_deg),
            "psi": float(psi_deg),
            "psi_deg": float(psi_deg),
            "psi_rad": float(psi),
            "psi_rel_deg": float(math.degrees(psi_rel_legacy)),
            "psi_rel_relkin_deg": float(math.degrees(psi_rel_relkin)),
            "psi_rel_minus_psi_root_deg": float(math.degrees(psi_rel_legacy - psi)),
            "vB_n_minus": float(vB_n),
            "vB_t_minus": float(vB_tn),
            "vb_n": float(vb_n),
            "vb_t": float(vb_tn),
            "vrel_n_minus": float(vrel_n_minus),
            "vrel_t_minus": float(vrel_t_minus),
            "Vn_plus": float(Vn_plus),
            "Vt_plus": float(Vt_plus),
            "Vn_plus_obs": float(Vn_plus),
            "Vt_plus_obs": float(Vt_plus),
            "vt_over_vn_abs": float(vt_over_vn_abs),
            "D": float(D_in),
            "D_in": float(D_in),
            "R_loc_in": float(R_loc),
            "Ft_denom": float(denom_ft),
            "Fw_denom": float(hitter.alpha * r_ball_ft),
            "raw_e_x": float(e_x_raw) if np.isfinite(e_x_raw) else np.nan,
            "e_x": float(e_x_raw) if np.isfinite(e_x_raw) else np.nan,
            "omega_plus": float(omega_plus) if np.isfinite(omega_plus) else np.nan,
            "regime": regime_label,
            "regime_label": regime_label,
            "log_prior_ex": float(log_prior_ex) if np.isfinite(log_prior_ex) else -np.inf,
            "d_n": float(d_n),
            "d_t": float(d_t),
        }

        if Vn_plus <= 0.0:
            bump("nonpositive_Vn_plus")
            out.update(branch_diag)
            continue
        if not (vt_over_vn_abs < 1.0):
            bump("vt_over_vn_ge_1")
            out.update(branch_diag)
            continue
        if d_n <= 0.0:
            bump("nonpositive_delta_n")
            out.update(branch_diag)
            continue
        if abs(D_in) > (R_loc + hitter.r_ball_in + 1e-8):
            bump("geometry_D_out_of_bounds")
            out.update(branch_diag)
            continue
        if abs(denom_ft) < DENOM_TOL:
            bump("bad_denom_Ft")
            out.update(branch_diag)
            continue
        if regime_label == "gross-slip":
            bump("regime_gross_slip")
            out.update(branch_diag)
            continue
        if (not np.isfinite(e_x_raw)) or (not (0.0 <= e_x_raw <= 0.6)):
            bump("e_x_out_of_bounds")
            out.update(branch_diag)
            continue
        if not np.isfinite(log_prior_ex):
            bump("nonfinite_log_prior_ex")
            out.update(branch_diag)
            continue

        out.update(branch_diag)
        out["root_exists"] = 1
        out["admissible"] = 1
        out["accepted_physics"] = 1
        out["failure_reason"] = "accepted"
        out["log_target"] = out["log_prior_x"] + out["log_prior_measurement"] + out["log_prior_e_y"] + out["log_prior_ex"]
        if not np.isfinite(out["log_target"]):
            out["accepted_physics"] = 0
            out["admissible"] = 0
            out["failure_reason"] = "nonfinite_log_target"
            out.update(classify_failure_stage(out))
            continue
        out.update(classify_failure_stage(out))
        return out

    out["failure_reason"] = max(fail_counter, key=fail_counter.get) if fail_counter else "no_admissible_root"
    out.update(classify_failure_stage(out))
    return out


def event_to_observed_dict(row: pd.Series, bank_module):
    """Build raw sensor-space observed dict."""
    out = bank_module.prepare_event_cache(row)
    out["event_id"] = int(row["event_id"])
    out["batter_name"] = str(row["batter_name"])
    return out


def _reflect_to_bounds(x: float, lo: float, hi: float) -> float:
    """Reflect scalar into closed interval."""
    y = float(x)
    while y < lo or y > hi:
        if y < lo:
            y = lo + (lo - y)
        if y > hi:
            y = hi - (y - hi)
    return float(np.clip(y, lo, hi))


def _scaled_sigmas(sensor_sigmas: dict[str, float], scales: dict[str, float]) -> dict[str, float]:
    """Scale measurement proposal sigmas."""
    out = dict(sensor_sigmas)
    g = float(scales.get("proposal_sd_measurement_block", 1.0))
    for k in out:
        out[k] = float(out[k] * g)
    out["hc_x"] *= float(scales.get("proposal_sd_phi", 1.0))
    out["hc_y"] *= float(scales.get("proposal_sd_phi", 1.0))
    out["attack_angle_deg"] *= float(scales.get("proposal_sd_attack_angle", 1.0))
    out["bat_speed_mph"] *= float(scales.get("proposal_sd_bat_speed", 1.0))
    return out


def propose_state(
    ev_obs: dict,
    hitter,
    calib_model: CalibModel,
    rng: np.random.Generator,
    bank_module,
    sensor_sigmas: dict[str, float],
    current_state: dict[str, Any] | None = None,
    proposal_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Propose one state under current scales."""
    proposal_scales = proposal_scales or {}
    lo, hi = hitter.bat_length - 11.0, hitter.bat_length
    if current_state is None or ("x" not in current_state):
        x = float(rng.uniform(lo, hi))
    else:
        sx = float(proposal_scales.get("proposal_sd_x", 0.9))
        x = _reflect_to_bounds(float(current_state["x"]) + float(rng.normal(0.0, sx)), lo, hi)

    sig = _scaled_sigmas(sensor_sigmas, proposal_scales)
    ev_star = bank_module.sample_measurement_star(ev_obs, rng=rng, sensor_sigmas=sig)
    logp_m = float(bank_module.log_p_mstar_given_mobs(ev_star, ev_obs, sensor_sigmas=sig))
    trans = build_transformed_measurements(ev_star, calib_model, bank_module)
    e_y_star, logp_ey = sample_e_y_star(x, hitter, bank_module, rng)
    solved = solve_exact_root_state(
        ev_obs=ev_obs,
        ev_star=ev_star,
        trans=trans,
        hitter=hitter,
        x_in=x,
        e_y_star=e_y_star,
        logp_measure=logp_m,
        logp_ey=logp_ey,
        bank_module=bank_module,
    )
    solved.update(
        {
            "event_id": int(ev_obs["event_id"]),
            "batter_name": str(ev_obs["batter_name"]),
            "phi_star": trans.get("phi_star_deg", np.nan),
            "delta_star": trans.get("delta_star_deg", np.nan),
            "d_tilde": trans.get("attack_direction_calib_deg", np.nan),
            "a_tilde": trans.get("attack_angle_calib_to_spray_deg", np.nan),
            "v_ss_tilde": trans.get("bat_speed_calib_to_spray_mph", np.nan),
            "attack_direction_obs_star": trans.get("attack_direction_obs_star_deg", np.nan),
            "attack_angle_obs_star": trans.get("attack_angle_obs_star_deg", np.nan),
            "bat_speed_obs_star": trans.get("bat_speed_obs_star_mph", np.nan),
            "spray_angle_obs_deg": np.nan,
            "attack_direction_obs_deg": float(ev_obs.get("attack_direction_deg", np.nan)),
            "attack_angle_obs_deg": float(ev_obs.get("attack_angle_deg", np.nan)),
            "bat_speed_obs_mph": float(ev_obs.get("bat_speed_mph", np.nan)),
            "exit_speed_obs_mph": float(ev_obs.get("launch_speed_mph", np.nan)),
            "launch_angle_obs_deg": float(ev_obs.get("launch_angle_deg", np.nan)),
            # Resampled launch angle from measurement layer (ev_star → trans); not spray angle / phi_star.
            "theta_star_deg": float(trans["theta_obs_deg"]),
            "log_q_proposal": float(-math.log(11.0) + logp_m + logp_ey),
            "failure_reason": solved.get("failure_reason", "accepted") or "accepted",
        }
    )
    return solved


def initialize_event_chain(
    ev_obs: dict,
    hitter,
    calib_model: CalibModel,
    rng: np.random.Generator,
    bank_module,
    sensor_sigmas: dict[str, float],
    max_init_attempts: int,
    proposal_scales: dict[str, float],
):
    """Initialize event chain with exact-root admissibility."""
    reasons: dict[str, int] = {}
    attempts = []
    for k in range(max_init_attempts):
        st = propose_state(
            ev_obs=ev_obs,
            hitter=hitter,
            calib_model=calib_model,
            rng=rng,
            bank_module=bank_module,
            sensor_sigmas=sensor_sigmas,
            current_state=None,
            proposal_scales=proposal_scales,
        )
        st["init_attempt_index"] = int(k)
        st["accepted_as_initial_state"] = False
        attempts.append(dict(st))
        if st.get("accepted_physics", 0) == 1 and np.isfinite(st.get("log_target", np.nan)):
            attempts[-1]["accepted_as_initial_state"] = True
            return st, reasons, pd.DataFrame(attempts)
        fr = str(st.get("failure_reason", "init_unknown"))
        reasons[fr] = reasons.get(fr, 0) + 1
    return None, reasons, pd.DataFrame(attempts)


def tune_proposals(
    ev_obs: dict,
    hitter,
    calib_model: CalibModel,
    curr: dict[str, Any],
    rng: np.random.Generator,
    bank_module,
    sensor_sigmas: dict[str, float],
    warmup_iters: int,
    target_low: float,
    target_high: float,
    proposal_scales: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Warmup-only proposal tuning to target acceptance range."""
    scales = dict(proposal_scales)
    win = 30
    acc, tot = 0, 0
    for wi in range(warmup_iters):
        prop = propose_state(
            ev_obs, hitter, calib_model, rng, bank_module, sensor_sigmas, current_state=curr, proposal_scales=scales
        )
        moved = 0
        if prop.get("accepted_physics", 0) == 1 and np.isfinite(prop.get("log_target", np.nan)):
            log_alpha = (prop["log_target"] - curr["log_target"]) + (curr.get("log_q_proposal", 0.0) - prop.get("log_q_proposal", 0.0))
            if np.log(rng.uniform()) < min(0.0, log_alpha):
                curr = prop
                moved = 1
        acc += moved
        tot += 1
        if (wi + 1) % win == 0:
            ar = acc / max(1, tot)
            mult = 1.0
            if ar < target_low:
                mult = 0.85
            elif ar > target_high:
                mult = 1.15
            for k in scales:
                scales[k] = float(np.clip(scales[k] * mult, 0.2, 3.0))
            acc = 0
            tot = 0
    return scales, curr


def run_event_mcmc(
    event_row: pd.Series,
    hitter,
    calib_model: CalibModel,
    rng: np.random.Generator,
    bank_module,
    sensor_sigmas: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run full event inference with init/tune/sample outputs."""
    ev_obs = event_to_observed_dict(event_row, bank_module)

    scales = {
        "proposal_sd_x": float(args.proposal_sd_x),
        "proposal_sd_measurement_block": float(args.proposal_sd_measurement_block),
        "proposal_sd_e_y": float(args.proposal_sd_e_y),
        "proposal_sd_phi": float(args.proposal_sd_phi),
        "proposal_sd_attack_angle": float(args.proposal_sd_attack_angle),
        "proposal_sd_bat_speed": float(args.proposal_sd_bat_speed),
    }

    init_state, init_reasons, init_attempts_df = initialize_event_chain(
        ev_obs=ev_obs,
        hitter=hitter,
        calib_model=calib_model,
        rng=rng,
        bank_module=bank_module,
        sensor_sigmas=sensor_sigmas,
        max_init_attempts=int(args.max_init_attempts),
        proposal_scales=scales,
    )
    init_attempts_df["init_retry_round"] = 0
    all_init_attempts = [init_attempts_df]
    merged_init_reasons = dict(init_reasons)

    if (
        init_state is None
        and bool(args.retry_failed_init)
        and int(args.max_init_attempts_retry) > int(args.max_init_attempts)
    ):
        retry_rounds = max(0, int(args.init_retry_rounds))
        retry_tries = int(args.max_init_attempts_retry)
        for rr in range(retry_rounds):
            st2, reasons2, init2 = initialize_event_chain(
                ev_obs=ev_obs,
                hitter=hitter,
                calib_model=calib_model,
                rng=rng,
                bank_module=bank_module,
                sensor_sigmas=sensor_sigmas,
                max_init_attempts=retry_tries,
                proposal_scales=scales,
            )
            init2["init_retry_round"] = int(rr + 1)
            all_init_attempts.append(init2)
            for k, v in reasons2.items():
                merged_init_reasons[k] = merged_init_reasons.get(k, 0) + int(v)
            if st2 is not None:
                init_state = st2
                break

    init_attempts_df = pd.concat(all_init_attempts, ignore_index=True)
    init_attempts_df["global_init_attempt_index"] = np.arange(len(init_attempts_df), dtype=int)

    if init_state is None:
        return {
            "event_id": int(ev_obs["event_id"]),
            "batter_name": str(ev_obs["batter_name"]),
            "status": "failed_init",
            "init_reason_counter_json": json.dumps(merged_init_reasons),
            "init_attempts_df": init_attempts_df,
            "proposal_trace_df": pd.DataFrame(),
            "posterior_draws_df": pd.DataFrame(),
            "tuned_scales": scales,
        }

    curr = dict(init_state)
    if args.tune_proposals and int(args.warmup_iters) > 0:
        scales, curr = tune_proposals(
            ev_obs=ev_obs,
            hitter=hitter,
            calib_model=calib_model,
            curr=curr,
            rng=rng,
            bank_module=bank_module,
            sensor_sigmas=sensor_sigmas,
            warmup_iters=int(args.warmup_iters),
            target_low=float(args.target_accept_low),
            target_high=float(args.target_accept_high),
            proposal_scales=scales,
        )

    trace_rows = []
    post_rows = []
    accepted_moves = 0
    n_iter = int(args.n_iter)
    burn = int(args.burn)
    thin = int(args.thin)
    stride = max(1, int(args.proposal_trace_stride))

    for it in range(n_iter):
        prop = propose_state(
            ev_obs, hitter, calib_model, rng, bank_module, sensor_sigmas, current_state=curr, proposal_scales=scales
        )
        ok_prop = prop.get("accepted_physics", 0) == 1 and np.isfinite(prop.get("log_target", np.nan))
        moved = 0
        if ok_prop:
            log_alpha = (prop["log_target"] - curr["log_target"]) + (curr.get("log_q_proposal", 0.0) - prop.get("log_q_proposal", 0.0))
            if np.log(rng.uniform()) < min(0.0, log_alpha):
                curr = prop
                moved = 1
                accepted_moves += 1

        prop["iter"] = int(it)
        prop["chain_id"] = 0
        prop["proposal_index"] = int(it)
        prop["current_or_proposed"] = "proposed"
        prop["accepted_move"] = int(moved)
        if moved == 1:
            prop["failure_reason"] = "accepted"
        prop.update(classify_failure_stage(prop))

        if args.save_full_proposal_trace and (it % stride == 0):
            trace_rows.append(dict(prop))

        if it >= burn and ((it - burn) % thin == 0):
            kept = dict(curr)
            kept["iter"] = int(it)
            kept["chain_id"] = 0
            kept["proposal_index"] = int(it)
            kept["accepted_draw"] = True
            kept["failure_reason"] = "accepted"
            kept.update(classify_failure_stage(kept))
            post_rows.append(kept)

    trace_df = pd.DataFrame(trace_rows)
    posterior_df = pd.DataFrame(post_rows)
    status = "ok" if len(posterior_df) else "empty_posterior"
    return {
        "event_id": int(ev_obs["event_id"]),
        "batter_name": str(ev_obs["batter_name"]),
        "status": status,
        "accept_rate": float(accepted_moves / max(1, n_iter)),
        "init_reason_counter_json": json.dumps(merged_init_reasons),
        "init_attempts_df": init_attempts_df,
        "proposal_trace_df": trace_df,
        "posterior_draws_df": posterior_df,
        "tuned_scales": scales,
    }


# ---------------------------------------------------------------------
# Summaries / quality / exports
# ---------------------------------------------------------------------
def summarize_event_proposals(trace_evt: pd.DataFrame) -> dict[str, Any]:
    """Proposal summary for one event."""
    if trace_evt.empty:
        return {}

    def _nan_stats(series_like):
        vals = pd.to_numeric(pd.Series(series_like), errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan, np.nan
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        return mean, sd

    raw_ex_mean, raw_ex_sd = _nan_stats(trace_evt.get("raw_e_x", np.nan))
    vtvn_series = trace_evt.get("abs_vt_over_vn", trace_evt.get("vt_over_vn_abs", np.nan))
    vtvn_mean, vtvn_sd = _nan_stats(vtvn_series)
    ft_mean, ft_sd = _nan_stats(trace_evt.get("Ft_denom", np.nan))
    ft_vals = pd.to_numeric(pd.Series(trace_evt.get("Ft_denom", np.nan)), errors="coerce").to_numpy(dtype=float)
    ft_vals = ft_vals[np.isfinite(ft_vals)]
    pct_ft_near_zero = float(np.mean(np.abs(ft_vals) < DENOM_TOL)) if len(ft_vals) else np.nan

    out = {
        "event_id": int(trace_evt["event_id"].iloc[0]),
        "batter_name": str(trace_evt["batter_name"].iloc[0]),
        "number_proposals": int(len(trace_evt)),
        "number_accepted_moves": int(trace_evt["accepted_move"].sum()) if "accepted_move" in trace_evt else 0,
        "acceptance_rate": float(trace_evt["accepted_move"].mean()) if "accepted_move" in trace_evt else np.nan,
        "root_existence_rate": float(trace_evt.get("root_exists_flag", trace_evt.get("root_exists", 0)).mean()),
        "root_in_support_rate": float(trace_evt.get("root_in_support_flag", trace_evt.get("root_exists", 0)).mean()),
        "admissibility_rate": float(trace_evt.get("admissible_flag", trace_evt.get("admissible", 0)).mean()),
        "mean_raw_e_x_proposal": raw_ex_mean,
        "sd_raw_e_x_proposal": raw_ex_sd,
        "mean_abs_vt_over_vn_proposal": vtvn_mean,
        "sd_abs_vt_over_vn_proposal": vtvn_sd,
        "mean_Ft_denom_proposal": ft_mean,
        "sd_Ft_denom_proposal": ft_sd,
        "pct_Ft_denom_near_zero_proposal": pct_ft_near_zero,
    }
    fr = trace_evt.get("failure_reason", pd.Series(["accepted"] * len(trace_evt))).fillna("accepted").replace("", "accepted")
    out["dominant_failure_reason"] = fr.value_counts().index[0] if len(fr) else "accepted"
    for k, v in fr.value_counts().items():
        out[f"failure_count_{str(k).replace(' ', '_')}"] = int(v)
    rg = trace_evt.get("regime_label", trace_evt.get("regime", pd.Series(["missing"] * len(trace_evt)))).fillna("missing")
    for k, v in rg.value_counts().items():
        out[f"regime_count_{str(k).replace(' ', '_')}"] = int(v)
    br = trace_evt.get("branch_index", trace_evt.get("branch_idx", pd.Series([-1] * len(trace_evt)))).fillna(-1)
    for k, v in br.value_counts().items():
        out[f"branch_count_{int(k)}"] = int(v)
    return out


def summarize_event_posterior(post_evt: pd.DataFrame) -> dict[str, Any]:
    """Accepted-only posterior summary for one event."""
    if post_evt.empty:
        return {}
    out: dict[str, Any] = {
        "event_id": int(post_evt["event_id"].iloc[0]),
        "batter_name": str(post_evt["batter_name"].iloc[0]),
        "number_posterior_draws": int(len(post_evt)),
    }

    def qstats(prefix: str, arr: np.ndarray) -> None:
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            for s in ["mean", "sd", "q05", "q25", "q50", "q75", "q95"]:
                out[f"{prefix}_{s}"] = np.nan
            return
        out[f"{prefix}_mean"] = float(np.mean(x))
        out[f"{prefix}_sd"] = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
        out[f"{prefix}_q05"] = float(np.quantile(x, 0.05))
        out[f"{prefix}_q25"] = float(np.quantile(x, 0.25))
        out[f"{prefix}_q50"] = float(np.quantile(x, 0.50))
        out[f"{prefix}_q75"] = float(np.quantile(x, 0.75))
        out[f"{prefix}_q95"] = float(np.quantile(x, 0.95))

    qstats("x", post_evt.get("x", np.array([])))
    qstats("psi", post_evt.get("psi", post_evt.get("psi_deg", np.array([]))))
    qstats("e_x", post_evt.get("e_x", np.array([])))
    qstats("e_y_star", post_evt.get("e_y_star", np.array([])))
    qstats("omega_plus", post_evt.get("omega_plus", np.array([])))

    vtvn = post_evt.get("abs_vt_over_vn", post_evt.get("vt_over_vn_abs", np.nan))
    out["accepted_only_mean_abs_vt_over_vn"] = float(np.nanmean(vtvn))
    out["accepted_only_sd_abs_vt_over_vn"] = float(np.nanstd(vtvn, ddof=1)) if len(post_evt) > 1 else np.nan
    out["accepted_only_mean_raw_e_x"] = float(np.nanmean(post_evt.get("raw_e_x", np.nan)))
    out["accepted_only_sd_raw_e_x"] = float(np.nanstd(post_evt.get("raw_e_x", np.nan), ddof=1)) if len(post_evt) > 1 else np.nan
    out["accepted_only_mean_Ft_denom"] = float(np.nanmean(post_evt.get("Ft_denom", np.nan)))
    out["accepted_only_sd_Ft_denom"] = float(np.nanstd(post_evt.get("Ft_denom", np.nan), ddof=1)) if len(post_evt) > 1 else np.nan

    rg = post_evt.get("regime_label", post_evt.get("regime", pd.Series(["missing"] * len(post_evt)))).fillna("missing")
    for k, v in rg.value_counts().items():
        out[f"accepted_regime_count_{str(k).replace(' ', '_')}"] = int(v)
    br = post_evt.get("branch_index", post_evt.get("branch_idx", pd.Series([-1] * len(post_evt)))).fillna(-1)
    for k, v in br.value_counts().items():
        out[f"accepted_branch_count_{int(k)}"] = int(v)

    x = pd.to_numeric(post_evt.get("x", np.nan), errors="coerce")
    if x.notna().any():
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        out["x_near_lower_support_rate"] = float(np.nanmean(x <= (lo + 0.2)))
        out["x_near_upper_support_rate"] = float(np.nanmean(x >= (hi - 0.2)))
    else:
        out["x_near_lower_support_rate"] = np.nan
        out["x_near_upper_support_rate"] = np.nan
    psi = pd.to_numeric(post_evt.get("psi", post_evt.get("psi_deg", np.nan)), errors="coerce")
    out["psi_near_support_edge_rate"] = float(np.nanmean(np.abs(psi) >= 65.0))
    ey = pd.to_numeric(post_evt.get("e_y_star", np.nan), errors="coerce")
    out["e_y_near_lower_bound_rate"] = float(np.nanmean(ey <= (EY_BOUNDS[0] + 0.02)))
    out["e_y_near_upper_bound_rate"] = float(np.nanmean(ey >= (EY_BOUNDS[1] - 0.02)))
    return out


def compute_event_quality(
    proposal_row: pd.Series | None,
    posterior_row: pd.Series | None,
    chain_row: pd.Series | None,
) -> dict[str, Any]:
    """Compute event quality class/score/weight."""
    acc = float(proposal_row.get("acceptance_rate", np.nan)) if proposal_row is not None else np.nan
    ess_x = float(chain_row.get("ess_x", np.nan)) if chain_row is not None else np.nan
    ess_psi = float(chain_row.get("ess_psi", np.nan)) if chain_row is not None else np.nan
    draw_n = int(posterior_row.get("number_posterior_draws", 0)) if posterior_row is not None else 0

    x_edge = float(max(posterior_row.get("x_near_lower_support_rate", 0.0), posterior_row.get("x_near_upper_support_rate", 0.0))) if posterior_row is not None else np.nan
    psi_edge = float(posterior_row.get("psi_near_support_edge_rate", np.nan)) if posterior_row is not None else np.nan
    ey_edge = float(max(posterior_row.get("e_y_near_lower_bound_rate", 0.0), posterior_row.get("e_y_near_upper_bound_rate", 0.0))) if posterior_row is not None else np.nan

    rej_flag = bool(np.isfinite(acc) and (acc < 0.01))
    poor_mix = bool((np.isfinite(ess_x) and ess_x < 50) or (np.isfinite(ess_psi) and ess_psi < 50))
    low_draw = bool(draw_n < 80)
    x_flag = bool(np.isfinite(x_edge) and x_edge > 0.25)
    psi_flag = bool(np.isfinite(psi_edge) and psi_edge > 0.25)
    ey_flag = bool(np.isfinite(ey_edge) and ey_edge > 0.25)

    acc_score = float(np.clip(acc / 0.2, 0.0, 1.0)) if np.isfinite(acc) else 0.0
    ess_score = float(np.clip(min(ess_x if np.isfinite(ess_x) else 0.0, ess_psi if np.isfinite(ess_psi) else 0.0) / 300.0, 0.0, 1.0))
    edge_pen = float(np.mean([x_flag, psi_flag, ey_flag]))
    draw_score = float(np.clip(draw_n / 250.0, 0.0, 1.0))
    quality_score = float(np.clip(0.35 * acc_score + 0.35 * ess_score + 0.20 * draw_score + 0.10 * (1.0 - edge_pen), 0.0, 1.0))

    if rej_flag or poor_mix:
        cls = "red"
    elif x_flag or psi_flag or ey_flag or low_draw:
        cls = "yellow"
    else:
        cls = "green"

    return {
        "acceptance_rate": acc,
        "ess_x": ess_x,
        "ess_psi": ess_psi,
        "rejection_rate_flag": rej_flag,
        "poor_mixing_flag": poor_mix,
        "low_draw_count_flag": low_draw,
        "x_near_support_flag": x_flag,
        "psi_near_support_flag": psi_flag,
        "e_y_near_bound_flag": ey_flag,
        "event_quality_score": quality_score,
        "event_reliability_weight": quality_score,
        "posterior_reliability_class": cls,
    }


def assign_splits(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    group_by: str,
) -> pd.DataFrame:
    """Assign train/val/test split labels with grouping."""
    out = df.copy()
    if len(out) == 0:
        out["split"] = pd.Series(dtype="object")
        return out

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train/val/test fractions must sum to 1")

    rng = np.random.default_rng(seed)
    if group_by == "hitter" and "batter_name" in out.columns:
        keys = out["batter_name"].astype(str).drop_duplicates().to_list()
        rng.shuffle(keys)
        n = len(keys)
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        tr = set(keys[:n_tr])
        va = set(keys[n_tr : n_tr + n_va])
        out["split"] = np.where(out["batter_name"].isin(tr), "train", np.where(out["batter_name"].isin(va), "val", "test"))
    else:
        keys = out["event_id"].drop_duplicates().to_list()
        rng.shuffle(keys)
        n = len(keys)
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        tr = set(keys[:n_tr])
        va = set(keys[n_tr : n_tr + n_va])
        out["split"] = np.where(out["event_id"].isin(tr), "train", np.where(out["event_id"].isin(va), "val", "test"))
    return out


def build_training_exports(
    posterior_draws: pd.DataFrame,
    selected_events: pd.DataFrame,
    quality_df: pd.DataFrame,
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build raw/basic/strict draw-level exports and train-ready tables."""
    train_dir = out_root / "train_exports"
    train_dir.mkdir(parents=True, exist_ok=True)
    if posterior_draws.empty:
        write_table(pd.DataFrame(), train_dir / "posterior_draws_long_raw.parquet", args.compression_codec)
        write_json(train_dir / "training_quality_summary.json", {"status": "no_draws"})
        return {"status": "no_draws"}

    ctx_cols = [
        "event_id",
        "batter_name",
        "pitch_type",
        "plate_x",
        "plate_z",
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "balls",
        "strikes",
        "stand",
        "p_throws",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
    ]
    ctx = selected_events[[c for c in ctx_cols if c in selected_events.columns]].drop_duplicates("event_id")
    df = posterior_draws.merge(ctx, on=["event_id", "batter_name"], how="left")
    df = df.merge(
        quality_df[
            [
                "event_id",
                "posterior_reliability_class",
                "event_quality_score",
                "event_reliability_weight",
                "x_near_support_flag",
                "psi_near_support_flag",
                "e_y_near_bound_flag",
                "poor_mixing_flag",
            ]
        ],
        on="event_id",
        how="left",
    )

    n_by_event = df.groupby("event_id").size().rename("n_draws_event")
    df = df.merge(n_by_event, on="event_id", how="left")
    df["uniform_draw_weight_within_event"] = 1.0 / df["n_draws_event"].clip(lower=1)
    if "log_target" in df.columns:
        df["normalized_log_target_weight"] = np.nan
        for eid, g in df.groupby("event_id"):
            lt = g["log_target"].to_numpy(dtype=float)
            m = np.nanmax(lt)
            w = np.exp(lt - m)
            s = np.nansum(w)
            if s > 0:
                df.loc[g.index, "normalized_log_target_weight"] = w / s
    else:
        df["normalized_log_target_weight"] = np.nan

    df["event_quality_weight"] = df["event_reliability_weight"].fillna(0.0)
    df["combined_training_weight"] = (
        0.4 * df["uniform_draw_weight_within_event"]
        + 0.3 * df["event_quality_weight"]
        + 0.3 * df["normalized_log_target_weight"].fillna(0.0)
    )

    # draw-level screening
    root_in_support = (
        pd.to_numeric(df["root_in_support_flag"], errors="coerce").fillna(1).astype(int).eq(1)
        if "root_in_support_flag" in df.columns
        else pd.Series(True, index=df.index)
    )
    admissible = (
        pd.to_numeric(df["admissible_flag"], errors="coerce").fillna(1).astype(int).eq(1)
        if "admissible_flag" in df.columns
        else pd.Series(True, index=df.index)
    )
    true_num = (
        df["true_numerical_failure"].fillna(False).astype(bool)
        if "true_numerical_failure" in df.columns
        else pd.Series(False, index=df.index)
    )

    df["draw_basic_screen_pass"] = (
        df["accepted_draw"].fillna(False).astype(bool)
        & root_in_support
        & admissible
        & (~true_num)
    )
    df["draw_strict_screen_pass"] = (
        df["draw_basic_screen_pass"]
        & df.get("e_x", np.nan).between(0.0, 1.0, inclusive="both")
        & df.get("abs_vt_over_vn", np.nan).fillna(np.inf).lt(1.0)
    )
    df["event_basic_screen_pass"] = ~df["posterior_reliability_class"].fillna("red").eq("red")
    df["event_strict_screen_pass"] = df["posterior_reliability_class"].fillna("red").eq("green")

    raw = assign_splits(df, args.train_frac, args.val_frac, args.test_frac, args.split_seed, args.group_splits_by)
    basic = raw[raw["draw_basic_screen_pass"]].copy()
    strict = raw[raw["draw_strict_screen_pass"] & raw["event_strict_screen_pass"]].copy()

    write_table(raw, train_dir / "posterior_draws_long_raw.parquet", args.compression_codec)
    write_table(basic, train_dir / "posterior_draws_long_basic_screened.parquet", args.compression_codec)
    write_table(strict, train_dir / "posterior_draws_long_strict_screened.parquet", args.compression_codec)

    # train-ready tables
    g_cols = [c for c in ctx_cols if c in raw.columns]
    u_cols = [c for c in ["a_tilde", "d_tilde", "v_ss_tilde", "v_coll", "vx_bat", "vy_bat", "vz_bat", "delta_star"] if c in raw.columns]
    z_cols = [c for c in ["x", "psi", "e_x", "e_y_star", "omega_plus"] if c in raw.columns]
    meta_cols = [
        c
        for c in [
            "event_id",
            "batter_name",
            "split",
            "branch_index",
            "regime_label",
            "log_target",
            "uniform_draw_weight_within_event",
            "event_quality_weight",
            "normalized_log_target_weight",
            "combined_training_weight",
            "draw_basic_screen_pass",
            "draw_strict_screen_pass",
            "event_basic_screen_pass",
            "event_strict_screen_pass",
            "posterior_reliability_class",
            "event_quality_score",
        ]
        if c in raw.columns
    ]

    ctx_lat = raw[[*meta_cols, *g_cols, *z_cols]].copy()
    ctx_up = raw[[*meta_cols, *g_cols, *u_cols]].copy()
    joint = raw[[*meta_cols, *g_cols, *u_cols, *z_cols]].copy()
    write_table(ctx_lat, train_dir / "context_to_latent_train.parquet", args.compression_codec)
    write_table(ctx_up, train_dir / "context_to_upstream_train.parquet", args.compression_codec)
    write_table(joint, train_dir / "context_upstream_latent_joint_train.parquet", args.compression_codec)

    # event-level targets
    psi_col = "psi" if "psi" in raw.columns else ("psi_deg" if "psi_deg" in raw.columns else None)
    agg_map = {
        "x_mean": ("x", "mean"),
        "x_q50": ("x", "median"),
        "e_x_mean": ("e_x", "mean"),
        "e_y_star_mean": ("e_y_star", "mean"),
        "omega_plus_mean": ("omega_plus", "mean"),
        "n_draws": ("event_id", "size"),
        "event_quality_score": ("event_quality_score", "mean"),
    }
    if psi_col is not None:
        agg_map["psi_mean"] = (psi_col, "mean")
        agg_map["psi_q50"] = (psi_col, "median")

    agg = (
        raw.groupby(["event_id", "batter_name", "split"], as_index=False)
        .agg(**agg_map)
    )
    write_table(agg, train_dir / "event_level_targets.parquet", args.compression_codec)

    split_manifest = {
        "group_splits_by": args.group_splits_by,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "split_seed": args.split_seed,
        "counts": raw["split"].value_counts().to_dict(),
    }
    write_json(train_dir / "split_manifest.json", split_manifest)

    feature_schema = {
        "context_features": g_cols,
        "upstream_features": u_cols,
        "latent_targets": z_cols,
        "categorical_columns": [c for c in ["batter_name", "pitch_type", "stand", "p_throws", "split", "regime_label"] if c in raw.columns],
        "continuous_columns": [c for c in raw.columns if c not in {"event_id", "batter_name"} and pd.api.types.is_numeric_dtype(raw[c])],
        "column_roles": {
            "observed_context": g_cols,
            "transformed_upstream": u_cols,
            "latent": z_cols,
            "weights": [c for c in ["uniform_draw_weight_within_event", "event_quality_weight", "normalized_log_target_weight", "combined_training_weight"] if c in raw.columns],
        },
        "units": {
            "x": "in",
            "psi": "deg",
            "e_x": "unitless",
            "e_y_star": "unitless",
            "omega_plus": "rad/s",
            "v_coll": "mph",
            "vx_bat": "mph",
            "vy_bat": "mph",
            "vz_bat": "mph",
        },
    }
    write_json(train_dir / "feature_schema.json", feature_schema)

    # preprocessing artifacts (simple standardization stats)
    stats = {}
    for c in [*g_cols, *u_cols, *z_cols]:
        if c in raw.columns and pd.api.types.is_numeric_dtype(raw[c]):
            vals = pd.to_numeric(raw[c], errors="coerce")
            stats[c] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    preprocessing = {
        "standardization_stats": stats,
        "screening_rules": {
            "basic": "accepted_draw && root_in_support && admissible && !true_numerical_failure",
            "strict": "basic && e_x in [0,1] && abs_vt_over_vn < 1 && event_strict_screen_pass",
        },
    }
    write_json(train_dir / "preprocessing_artifacts.json", preprocessing)
    write_json(
        train_dir / "training_quality_summary.json",
        {
            "n_raw": int(len(raw)),
            "n_basic": int(len(basic)),
            "n_strict": int(len(strict)),
            "quality_class_counts": raw["posterior_reliability_class"].value_counts(dropna=False).to_dict(),
        },
    )
    return {"n_raw": len(raw), "n_basic": len(basic), "n_strict": len(strict)}


def build_calibration_exports(
    posterior_draws: pd.DataFrame,
    selected_events: pd.DataFrame,
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build calibration/SBI-facing exports and simulator hooks."""
    cal_dir = out_root / "calibration_exports"
    cal_dir.mkdir(parents=True, exist_ok=True)
    if posterior_draws.empty:
        write_table(pd.DataFrame(), cal_dir / "observed_inference_pack.parquet", args.compression_codec)
        write_json(cal_dir / "calibration_manifest.json", {"status": "no_draws"})
        return {"status": "no_draws"}

    y_cols = [c for c in ["exit_speed_obs_mph", "launch_angle_obs_deg", "spray_angle_obs_deg"] if c in posterior_draws.columns]
    g_cols = [c for c in ["event_id", "batter_name", "plate_x", "plate_z", "release_speed", "release_spin_rate", "spin_axis", "balls", "strikes", "stand", "p_throws"] if c in selected_events.columns]
    ctx = selected_events[g_cols].drop_duplicates("event_id")
    pack = posterior_draws.merge(ctx, on=["event_id", "batter_name"], how="left")
    write_table(pack, cal_dir / "observed_inference_pack.parquet", args.compression_codec)

    # keep explicit missing simulator hooks
    write_json(
        cal_dir / "sbc_ready_schema.json",
        {
            "required_simulator_inputs": ["context_g", "latent_theta_or_z"],
            "required_simulator_outputs": ["simulated_observables_y"],
            "notes": "Forward simulator not wired in this production script; hooks are prepared.",
        },
    )
    write_json(
        cal_dir / "calibration_manifest.json",
        {
            "observed_inference_pack_rows": int(len(pack)),
            "ppc_ready": False,
            "sbc_ready": False,
            "todo": [
                "Implement forward simulator adapter",
                "Implement posterior predictive generator",
                "Implement SBC rank/coverage pipeline",
            ],
        },
    )
    write_table(pd.DataFrame(), cal_dir / "posterior_predictive_draws.parquet", args.compression_codec)
    write_table(pd.DataFrame(), cal_dir / "posterior_predictive_summary.parquet", args.compression_codec)
    write_table(pd.DataFrame(), cal_dir / "calibration_failures.parquet", args.compression_codec)
    write_table(pd.DataFrame(), cal_dir / "sbc_simulations.parquet", args.compression_codec)
    write_table(pd.DataFrame(), cal_dir / "sbc_rank_statistics.parquet", args.compression_codec)
    write_table(pd.DataFrame(), cal_dir / "sbc_coverage_summary.parquet", args.compression_codec)
    return {"observed_inference_pack_rows": len(pack)}


def aggregate_outputs(
    all_init: list[pd.DataFrame],
    all_trace: list[pd.DataFrame],
    all_post: list[pd.DataFrame],
    all_prop_diag: list[dict[str, Any]],
    all_post_sum: list[dict[str, Any]],
    all_chain_diag: list[dict[str, Any]],
    all_quality: list[dict[str, Any]],
    tuned_rows: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Aggregate per-event outputs into combined tables."""
    init_df = pd.concat(all_init, ignore_index=True) if all_init else pd.DataFrame()
    trace_df = pd.concat(all_trace, ignore_index=True) if all_trace else pd.DataFrame()
    post_df = pd.concat(all_post, ignore_index=True) if all_post else pd.DataFrame()
    prop_diag_df = pd.DataFrame(all_prop_diag)
    post_sum_df = pd.DataFrame(all_post_sum)
    chain_df = pd.DataFrame(all_chain_diag)
    quality_df = pd.DataFrame(all_quality)
    tuned_df = pd.DataFrame(tuned_rows)

    if not trace_df.empty:
        fr_counts = trace_df["failure_reason"].fillna("missing").replace("", "accepted").value_counts().rename_axis("failure_reason").reset_index(name="count")
        fs_counts = (
            pd.DataFrame(
                {
                    "failed_stage_1_root": [int(trace_df.get("failed_stage_1_root", False).sum())],
                    "failed_stage_2_support": [int(trace_df.get("failed_stage_2_support", False).sum())],
                    "failed_stage_3_admissibility": [int(trace_df.get("failed_stage_3_admissibility", False).sum())],
                    "failed_stage_4_downstream": [int(trace_df.get("failed_stage_4_downstream", False).sum())],
                    "true_numerical_failure": [int(trace_df.get("true_numerical_failure", False).sum())],
                }
            )
            .melt(var_name="stage_flag", value_name="count")
            .sort_values("stage_flag")
            .reset_index(drop=True)
        )
    else:
        fr_counts = pd.DataFrame(columns=["failure_reason", "count"])
        fs_counts = pd.DataFrame(columns=["stage_flag", "count"])

    if not trace_df.empty:
        psi_rows = []
        for (eid, bn), g in trace_df.groupby(["event_id", "batter_name"]):
            p0 = pd.to_numeric(g.get("psi_root_deg", g.get("psi", np.nan)), errors="coerce")
            p1 = pd.to_numeric(g.get("psi_rel_deg", np.nan), errors="coerce")
            p2 = pd.to_numeric(g.get("psi_rel_relkin_deg", np.nan), errors="coerce")
            d1 = p1 - p0
            d2 = p2 - p0
            psi_rows.append(
                {
                    "event_id": int(eid),
                    "batter_name": str(bn),
                    "psi_root_mean": float(np.nanmean(p0)),
                    "psi_rel_mean": float(np.nanmean(p1)),
                    "psi_rel_relkin_mean": float(np.nanmean(p2)),
                    "mean_gap_psi_rel_minus_root": float(np.nanmean(d1)),
                    "sd_gap_psi_rel_minus_root": float(np.nanstd(d1, ddof=1)) if d1.notna().sum() > 1 else np.nan,
                    "mean_gap_psi_rel_relkin_minus_root": float(np.nanmean(d2)),
                    "sd_gap_psi_rel_relkin_minus_root": float(np.nanstd(d2, ddof=1)) if d2.notna().sum() > 1 else np.nan,
                    "frac_large_gap_psi_rel_minus_root": float(np.nanmean(np.abs(d1) > 45.0)),
                    "frac_large_gap_psi_rel_relkin_minus_root": float(np.nanmean(np.abs(d2) > 45.0)),
                }
            )
        psi_df = pd.DataFrame(psi_rows)
    else:
        psi_df = pd.DataFrame()

    if not trace_df.empty:
        if "abs_vt_over_vn" not in trace_df.columns and "vt_over_vn_abs" in trace_df.columns:
            trace_df["abs_vt_over_vn"] = trace_df["vt_over_vn_abs"]
        tang_prop = trace_df.groupby(["event_id", "batter_name"], as_index=False).agg(
            raw_e_x_mean_proposal=("raw_e_x", "mean"),
            raw_e_x_sd_proposal=("raw_e_x", "std"),
            Ft_denom_mean_proposal=("Ft_denom", "mean"),
            Ft_denom_sd_proposal=("Ft_denom", "std"),
            abs_vt_over_vn_mean_proposal=("abs_vt_over_vn", "mean"),
            abs_vt_over_vn_sd_proposal=("abs_vt_over_vn", "std"),
        )
    else:
        tang_prop = pd.DataFrame()

    if not post_df.empty:
        if "abs_vt_over_vn" not in post_df.columns and "vt_over_vn_abs" in post_df.columns:
            post_df["abs_vt_over_vn"] = post_df["vt_over_vn_abs"]
        tang_post = post_df.groupby(["event_id", "batter_name"], as_index=False).agg(
            raw_e_x_mean_posterior=("raw_e_x", "mean"),
            raw_e_x_sd_posterior=("raw_e_x", "std"),
            Ft_denom_mean_posterior=("Ft_denom", "mean"),
            Ft_denom_sd_posterior=("Ft_denom", "std"),
            abs_vt_over_vn_mean_posterior=("abs_vt_over_vn", "mean"),
            abs_vt_over_vn_sd_posterior=("abs_vt_over_vn", "std"),
        )
    else:
        tang_post = pd.DataFrame()

    return {
        "init_attempts": init_df,
        "proposal_trace": trace_df,
        "posterior_draws": post_df,
        "proposal_diagnostics_by_event": prop_diag_df,
        "posterior_summary_by_event": post_sum_df,
        "chain_diagnostics": chain_df,
        "event_quality_flags": quality_df,
        "tuned_proposal_scales_by_event": tuned_df,
        "failure_reason_counts": fr_counts,
        "failure_stage_counts": fs_counts,
        "psi_angle_audit": psi_df,
        "tangential_proposal_diagnostics": tang_prop,
        "tangential_posterior_diagnostics": tang_post,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    """Main production entrypoint."""
    args = build_arg_parser().parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    per_event_root = out_root / "per_event"
    agg_root = out_root / "aggregates"
    per_event_root.mkdir(parents=True, exist_ok=True)
    agg_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    # load source module
    spec = importlib.util.spec_from_file_location("mcmc_bank_prod", Path(args.source_script_path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sensor_sigmas = m.default_sensor_sigmas()

    # load/filter/select
    base_df, bip_model, hitters_by_name = load_and_filter_events(Path(args.input_data_path), m)
    calib_models, calib_diag = build_calibration_models(bip_model)
    event_ids = parse_event_ids(args.event_ids)
    selected = select_events(bip_model, event_ids, int(args.max_events), int(args.random_seed))

    # run metadata
    config_snapshot = vars(args).copy()
    config_snapshot.update(
        {
            "PSI_SUPPORT_DEG": PSI_SUPPORT_DEG,
            "EY_BOUNDS": EY_BOUNDS,
            "model_constraints": {
                "exact_root_only": True,
                "sampled_e_y_star": True,
                "d_tilde_eq_phi_star": True,
                "v_coll_formula": "v_ss_tilde*(x-6)/(L-12)",
                "no_soft_residual_fallback": True,
            },
        }
    )
    write_json(out_root / "config_snapshot.json", config_snapshot)
    write_json(
        out_root / "run_manifest.json",
        {
            "selected_events": int(len(selected)),
            "source_notebook": "/home/evangoforth03/Bayesian Research/MCMC 2/MCMC_Refactor_ExactRoot_RHH.ipynb",
            "source_script": str(args.source_script_path),
        },
    )
    (out_root / "run_log.txt").write_text("Production run started\n")

    write_table(selected, out_root / "selected_events.parquet", args.compression_codec)
    write_table(selected, out_root / "selected_events.csv")

    # resume status
    status_csv = out_root / "event_run_status.csv"
    status_parquet = out_root / "event_run_status.parquet"
    if args.resume and status_csv.exists():
        status_df = pd.read_csv(status_csv)
    else:
        status_df = pd.DataFrame(
            columns=[
                "event_id",
                "batter_name",
                "init_completed",
                "chain_completed",
                "outputs_written",
                "posterior_summary_written",
                "diagnostics_written",
                "quality_flag",
                "error_message",
                "runtime_seconds",
            ]
        )

    completed_ids = set()
    if args.resume and (not status_df.empty) and ("event_id" in status_df.columns):
        done = status_df[
            status_df["chain_completed"].fillna(False).astype(bool)
            & status_df["outputs_written"].fillna(False).astype(bool)
        ]
        completed_ids = set(pd.to_numeric(done["event_id"], errors="coerce").dropna().astype(int).tolist())

    all_init = []
    all_trace = []
    all_post = []
    all_prop_diag = []
    all_post_sum = []
    all_chain_diag = []
    all_quality = []
    tuned_rows = []
    run_start_ts = pd.Timestamp.utcnow()
    total_events = int(len(selected))
    processed_events = 0
    ok_chain_count = 0
    failed_init_count = 0

    for _, row in selected.iterrows():
        eid = int(row["event_id"])
        bname = str(row["batter_name"])
        if (eid in completed_ids) and (not args.overwrite):
            processed_events += 1
            elapsed = float((pd.Timestamp.utcnow() - run_start_ts).total_seconds())
            avg = elapsed / max(1, processed_events)
            eta = avg * max(0, total_events - processed_events)
            bar = progress_bar(processed_events, total_events)
            pct = 100.0 * processed_events / max(1, total_events)
            log(
                f"{bar} {processed_events}/{total_events} ({pct:5.1f}%) "
                f"elapsed={_fmt_hms(elapsed)} eta={_fmt_hms(eta)} "
                f"ok_chain={ok_chain_count} failed_init={failed_init_count} "
                f"[resume-skip event {eid}]"
            )
            continue

        t0 = pd.Timestamp.utcnow()
        stat_row = {
            "event_id": eid,
            "batter_name": bname,
            "init_completed": False,
            "chain_completed": False,
            "outputs_written": False,
            "posterior_summary_written": False,
            "diagnostics_written": False,
            "quality_flag": "red",
            "error_message": "",
            "runtime_seconds": np.nan,
        }
        try:
            hitter = hitters_by_name.get(bname)
            calib_model = calib_models.get(bname)
            if hitter is None or calib_model is None:
                raise RuntimeError("missing_hitter_or_calib")

            result = run_event_mcmc(row, hitter, calib_model, rng, m, sensor_sigmas, args)
            init_df = result["init_attempts_df"]
            trace_df = result["proposal_trace_df"]
            post_df = result["posterior_draws_df"]
            tuned = {"event_id": eid, "batter_name": bname, **result["tuned_scales"]}
            tuned_rows.append(tuned)

            # per-event outputs
            ev_dir = per_event_root / f"event_{eid}"
            ev_dir.mkdir(parents=True, exist_ok=True)
            if args.save_per_event_parquet:
                write_table(init_df, ev_dir / "init_attempts.parquet", args.compression_codec)
                if args.save_full_proposal_trace:
                    write_table(trace_df, ev_dir / "proposal_trace.parquet", args.compression_codec)
                write_table(post_df, ev_dir / "posterior_draws.parquet", args.compression_codec)
            else:
                init_df.to_csv(ev_dir / "init_attempts.csv", index=False)
                if args.save_full_proposal_trace:
                    trace_df.to_csv(ev_dir / "proposal_trace.csv", index=False)
                post_df.to_csv(ev_dir / "posterior_draws.csv", index=False)

            prop_diag = summarize_event_proposals(trace_df if not trace_df.empty else pd.DataFrame(
                [
                    {
                        "event_id": eid,
                        "batter_name": bname,
                        "failure_reason": "failed_init",
                        "accepted_move": 0,
                        "root_exists_flag": 0,
                        "root_in_support_flag": 0,
                        "admissible_flag": 0,
                    }
                ]
            ))
            post_sum = summarize_event_posterior(post_df)
            chain = {
                "event_id": eid,
                "batter_name": bname,
                "acceptance_rate": float(result.get("accept_rate", np.nan)),
                "mean_jump_size_x": float(np.nanmean(np.abs(np.diff(post_df["x"])))) if ("x" in post_df and len(post_df) > 1) else np.nan,
                "mean_jump_size_psi": float(np.nanmean(np.abs(np.diff(post_df.get("psi", post_df.get("psi_deg", np.nan)))))) if len(post_df) > 1 else np.nan,
                "lag1_autocorr_x": _safe_lag1(post_df["x"].to_numpy()) if "x" in post_df else np.nan,
                "lag1_autocorr_psi": _safe_lag1(post_df.get("psi", post_df.get("psi_deg", pd.Series(dtype=float))).to_numpy()) if len(post_df) else np.nan,
                "lag1_autocorr_e_x": _safe_lag1(post_df["e_x"].to_numpy()) if "e_x" in post_df else np.nan,
                "lag1_autocorr_e_y_star": _safe_lag1(post_df["e_y_star"].to_numpy()) if "e_y_star" in post_df else np.nan,
                "lag1_autocorr_omega_plus": _safe_lag1(post_df["omega_plus"].to_numpy()) if "omega_plus" in post_df else np.nan,
                "posterior_draw_count": int(len(post_df)),
            }
            chain["ess_x"] = _ess_from_rho1(chain["posterior_draw_count"], chain["lag1_autocorr_x"])
            chain["ess_psi"] = _ess_from_rho1(chain["posterior_draw_count"], chain["lag1_autocorr_psi"])
            chain["ess_e_x"] = _ess_from_rho1(chain["posterior_draw_count"], chain["lag1_autocorr_e_x"])
            chain["ess_e_y_star"] = _ess_from_rho1(chain["posterior_draw_count"], chain["lag1_autocorr_e_y_star"])
            chain["ess_omega_plus"] = _ess_from_rho1(chain["posterior_draw_count"], chain["lag1_autocorr_omega_plus"])

            q = compute_event_quality(
                proposal_row=pd.Series(prop_diag) if prop_diag else None,
                posterior_row=pd.Series(post_sum) if post_sum else None,
                chain_row=pd.Series(chain),
            )
            qrow = {"event_id": eid, "batter_name": bname, **q}

            # write event-level json snapshots
            write_json(ev_dir / "posterior_summary.json", post_sum if post_sum else {"status": "no_posterior_draws"})
            write_json(ev_dir / "chain_diagnostics.json", chain)
            write_json(ev_dir / "quality_flags.json", qrow)

            stat_row["init_completed"] = bool(result["status"] != "failed_init")
            stat_row["chain_completed"] = bool(result["status"] == "ok" or result["status"] == "empty_posterior")
            stat_row["outputs_written"] = True
            stat_row["posterior_summary_written"] = True
            stat_row["diagnostics_written"] = True
            stat_row["quality_flag"] = q["posterior_reliability_class"]
            if result["status"] == "failed_init":
                stat_row["error_message"] = f"failed_init: {result.get('init_reason_counter_json', '{}')}"

            # aggregate in-memory
            all_init.append(init_df)
            if args.save_full_proposal_trace:
                all_trace.append(trace_df)
            all_post.append(post_df)
            if prop_diag:
                all_prop_diag.append(prop_diag)
            if post_sum:
                all_post_sum.append(post_sum)
            all_chain_diag.append(chain)
            all_quality.append(qrow)

        except Exception as exc:  # pylint: disable=broad-except
            stat_row["error_message"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=1).strip()}"
        finally:
            dt = pd.Timestamp.utcnow() - t0
            stat_row["runtime_seconds"] = float(dt.total_seconds())
            status_df = pd.concat([status_df[status_df["event_id"] != eid], pd.DataFrame([stat_row])], ignore_index=True)
            write_table(status_df, status_parquet, args.compression_codec)
            write_table(status_df, status_csv)
            processed_events += 1
            if bool(stat_row["chain_completed"]):
                ok_chain_count += 1
            if not bool(stat_row["init_completed"]):
                failed_init_count += 1
            elapsed = float((pd.Timestamp.utcnow() - run_start_ts).total_seconds())
            avg = elapsed / max(1, processed_events)
            eta = avg * max(0, total_events - processed_events)
            bar = progress_bar(processed_events, total_events)
            pct = 100.0 * processed_events / max(1, total_events)
            log(
                f"{bar} {processed_events}/{total_events} ({pct:5.1f}%) "
                f"elapsed={_fmt_hms(elapsed)} eta={_fmt_hms(eta)} "
                f"ok_chain={ok_chain_count} failed_init={failed_init_count} "
                f"[event {eid} chain_completed={stat_row['chain_completed']} quality={stat_row['quality_flag']}]"
            )

    # aggregate outputs
    agg = aggregate_outputs(all_init, all_trace, all_post, all_prop_diag, all_post_sum, all_chain_diag, all_quality, tuned_rows)

    # combined outputs
    if args.save_combined_parquet:
        write_table(agg["proposal_diagnostics_by_event"], agg_root / "proposal_diagnostics_by_event.parquet", args.compression_codec)
        write_table(agg["posterior_summary_by_event"], agg_root / "posterior_summary_by_event.parquet", args.compression_codec)
        write_table(agg["chain_diagnostics"], agg_root / "chain_diagnostics.parquet", args.compression_codec)
        write_table(agg["event_quality_flags"], agg_root / "event_quality_flags.parquet", args.compression_codec)
        write_table(agg["psi_angle_audit"], agg_root / "psi_angle_audit.parquet", args.compression_codec)
        write_table(agg["tangential_proposal_diagnostics"], agg_root / "tangential_proposal_diagnostics.parquet", args.compression_codec)
        write_table(agg["tangential_posterior_diagnostics"], agg_root / "tangential_posterior_diagnostics.parquet", args.compression_codec)
        write_table(agg["failure_reason_counts"], agg_root / "failure_reason_counts.parquet", args.compression_codec)
        write_table(agg["failure_stage_counts"], agg_root / "failure_stage_counts.parquet", args.compression_codec)
        write_table(agg["tuned_proposal_scales_by_event"], agg_root / "tuned_proposal_scales_by_event.parquet", args.compression_codec)
        write_table(selected, agg_root / "context_event_table.parquet", args.compression_codec)

    if args.save_csv_summaries:
        write_table(agg["proposal_diagnostics_by_event"], agg_root / "proposal_diagnostics_by_event.csv")
        write_table(agg["posterior_summary_by_event"], agg_root / "posterior_summary_by_event.csv")
        write_table(agg["chain_diagnostics"], agg_root / "chain_diagnostics.csv")
        write_table(agg["event_quality_flags"], agg_root / "event_quality_flags.csv")

    # train exports
    if args.make_train_exports:
        train_meta = build_training_exports(
            posterior_draws=agg["posterior_draws"],
            selected_events=selected,
            quality_df=agg["event_quality_flags"],
            out_root=out_root,
            args=args,
        )
        log(f"train_exports: {train_meta}")

    # calibration exports
    if args.make_calibration_exports:
        cal_meta = build_calibration_exports(
            posterior_draws=agg["posterior_draws"],
            selected_events=selected,
            out_root=out_root,
            args=args,
        )
        log(f"calibration_exports: {cal_meta}")

    # final summary
    ok_init = int(status_df["init_completed"].fillna(False).astype(bool).sum()) if "init_completed" in status_df else 0
    ok_chain = int(status_df["chain_completed"].fillna(False).astype(bool).sum()) if "chain_completed" in status_df else 0
    acc_series = agg["proposal_diagnostics_by_event"]["acceptance_rate"] if not agg["proposal_diagnostics_by_event"].empty else pd.Series(dtype=float)
    quality_counts = agg["event_quality_flags"]["posterior_reliability_class"].value_counts(dropna=False).to_dict() if not agg["event_quality_flags"].empty else {}
    log("\n=== Production run summary ===")
    log(f"events successfully initialized: {ok_init}")
    log(f"events with completed chains: {ok_chain}")
    if len(acc_series):
        log(f"acceptance summary: {acc_series.describe(percentiles=[0.05,0.5,0.95]).to_dict()}")
    log(f"quality counts: {quality_counts}")


if __name__ == "__main__":
    main()
