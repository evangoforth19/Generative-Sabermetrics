"""
Deterministic forward physics decoder: (u, z, g, p) -> (EV, LA, SA) + nuisances.

Aligned with `run_inverse_collision_production.solve_exact_root_state` kinematics and
`run_mcmc_posterior_bank` geometry helpers. Stage-z `psi_deg` builds nhat/that;
`theta_deg` sets outgoing direction in the audited (r_hat(phi), e_z) plane; phi = d_tilde.

Normal outgoing speed uses the production normal constraint (F_n = 0) at fixed psi:
  V_n_plus = v_B_n + lambda * Delta_n,  lambda = (1 + e_y_star) / (1 + r_y(x)).

Then s, V_t_plus follow V_n_plus = s cos(theta - psi), V_t_plus = s sin(theta - psi).
Never form the tangent of (psi minus theta) in reversed order; never use psi_rel_* as the contact normal angle for nhat/that.

**Inputs:** ``decode_bip_from_sample`` calls ``validate_decoder_inputs`` first; see ``physics_decoder_contract`` and ``manifests/physics_decoder_input_contract.json``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .physics_decoder_contract import (
    DecoderInputError,
    DecoderInputs,
    DecoderOutputs,
    decoder_z_branch,
    validate_decoder_inputs,
)

# ---------------------------------------------------------------------------
# Load production modules (same repo; no package install required)
# ---------------------------------------------------------------------------

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))


def _bank_module():
    name = "_sbi_loaded_run_mcmc_posterior_bank"
    spec = importlib.util.spec_from_file_location(
        name,
        _MMC2_ROOT / "run_mcmc_posterior_bank.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _inverse_collision_module():
    name = "_sbi_loaded_run_inverse_collision_production"
    spec = importlib.util.spec_from_file_location(
        name,
        _MMC2_ROOT / "run_inverse_collision_production.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BANK = None
_IC = None


def get_bank_module():
    global _BANK
    if _BANK is None:
        _BANK = _bank_module()
    return _BANK


def get_inverse_collision_module():
    global _IC
    if _IC is None:
        _IC = _inverse_collision_module()
    return _IC


def hitter_from_p_dict(batter_name: str, p: dict[str, Any]):
    bank = get_bank_module()
    return bank.Hitter(
        name=str(batter_name),
        bat_length=float(p["bat_length_in"]),
        bat_weight=float(p["bat_weight_oz"]),
        x_cm_fixed=float(p["x_cm_fixed_in"]),
        r_g_fixed=float(p["r_g_fixed_in"]),
        r_ball_in=bank.BALL_RADIUS_IN,
        alpha=bank.ALPHA,
        Iz_oz_in2=float(p["Iz_oz_in2"]),
        ball_mass_oz=bank.BALL_MASS_OZ,
    )


def _spin_axis_deg_from_g(g: dict[str, Any]) -> float:
    if "spin_axis_deg" in g and np.isfinite(float(g["spin_axis_deg"])):
        return float(g["spin_axis_deg"])
    if "spin_axis_sin" in g and "spin_axis_cos" in g:
        return float(
            np.degrees(
                np.arctan2(float(g["spin_axis_sin"]), float(g["spin_axis_cos"]))
            )
        )
    raise KeyError("g must provide spin_axis_deg or (spin_axis_sin, spin_axis_cos)")


def build_transient_trans_and_pitch(
    u: dict[str, float],
    g: dict[str, Any],
    bank_module: Any,
):
    """
    Minimal `trans` + pitch for bat vector and omega_minus (inverse-collision lineage).

    Callers must satisfy `validate_decoder_inputs` on the full `DecoderInputs` before
    relying on this helper; it assumes decoder **g** contract (including `release_pos_y`).
    """
    phi_rad = math.radians(float(u["d_tilde"]))
    pitch = bank_module.build_incoming_pitch_from_sensor(
        vx0=float(g["vx0"]),
        vy0=float(g["vy0"]),
        vz0=float(g["vz0"]),
        ax=float(g["ax"]),
        ay=float(g["ay"]),
        az=float(g["az"]),
        release_pos_y=float(g["release_pos_y"]),
    )
    spin_axis_deg = _spin_axis_deg_from_g(g)
    release_rpm = float(g["release_spin_rate"])
    t_c = pitch["t_contact"]
    omega_minus = bank_module.relevant_omega_minus_from_sensor(
        release_spin_rate_rpm=release_rpm,
        spin_axis_deg=spin_axis_deg,
        phi_rad=phi_rad,
        t_contact=t_c,
        asr=float(g.get("active_spin_ratio", bank_module.ASR_DEFAULT)),
    )
    trans = {
        "phi_star_deg": float(u["d_tilde"]),
        "attack_angle_calib_to_spray_deg": float(u["a_tilde"]),
        "bat_speed_calib_to_spray_mph": float(u["v_ss_tilde"]),
        "omega_minus": float(omega_minus),
    }
    return trans, pitch


def decode_bip_analytic_ex(
    inp: DecoderInputs,
    *,
    admissibility: Literal["strict", "lenient"] = "strict",
) -> DecoderOutputs:
    """
    Decoder branch when stage-z supplies ``(x, e_y_star, psi_deg, e_x)`` (no ``theta_deg``).

    Uses the same normal closure ``V_n_plus`` as production, then tangential speed from the
    sampled tangential restitution ``e_x`` (inverse-collision algebra), then
    ``EV = ||(V_n_plus, V_t_plus)||`` and ``theta = psi + atan2(V_t_plus, V_n_plus)`` for ``LA``.
    """
    bank = get_bank_module()
    ic = get_inverse_collision_module()
    PSI_SUPPORT_DEG = ic.PSI_SUPPORT_DEG
    DENOM_TOL = ic.DENOM_TOL
    DEFAULT_MU = bank.DEFAULT_MU
    DEFAULT_G2 = bank.DEFAULT_G2_BASEBALL
    EPS_DT = 1e-8

    def _fail(reason: str) -> DecoderOutputs:
        return DecoderOutputs(
            ev_mph=float("nan"),
            launch_angle_deg=float("nan"),
            spray_angle_deg=float("nan"),
            e_x=None,
            omega_plus_rad_s=None,
            admissible=False,
            failure_reason=reason,
        )

    u, z, g, p = inp.u, inp.z, inp.g, inp.p
    try:
        hitter = hitter_from_p_dict(inp.batter_name, p)
        trans, pitch = build_transient_trans_and_pitch(u, g, bank)
    except Exception as ex:
        return _fail(f"input_parse:{type(ex).__name__}")

    x_in = float(z["x"])
    psi_deg = float(z["psi_deg"])
    e_y_star = float(z["e_y_star"])
    e_x_in = float(z["e_x"])
    phi_star_deg = float(u["d_tilde"])

    if not (PSI_SUPPORT_DEG[0] <= psi_deg <= PSI_SUPPORT_DEG[1]):
        if admissibility == "strict":
            return _fail("psi_outside_support")

    if not np.isfinite(psi_deg) or not np.isfinite(e_y_star) or not np.isfinite(e_x_in):
        return _fail("nonfinite_z_fields")

    L_in = float(hitter.bat_length)
    if not (L_in - 11.0 - 1e-9 <= x_in <= L_in + 1e-9):
        if admissibility == "strict":
            return _fail("x_outside_bat_support")

    phi_rad = math.radians(float(u["d_tilde"]))
    psi_rad = math.radians(psi_deg)

    try:
        bvec = ic.bat_vector_from_x(x_in, trans, hitter, bank)
    except Exception:
        return _fail("bat_vector_failure")

    vB_x = float(pitch["vin_x"])
    vB_y = float(pitch["vin_y"])
    vB_z = float(pitch["vin_z"])
    vB_r = vB_x * math.sin(phi_rad) + vB_y * math.cos(phi_rad)
    vB_t = vB_x * math.cos(phi_rad) - vB_y * math.sin(phi_rad)

    vb_x = float(bvec["vx_bat_mph"]) * bank.MPH_TO_FPS
    vb_y = float(bvec["vy_bat_mph"]) * bank.MPH_TO_FPS
    vb_z = float(bvec["vz_bat_mph"]) * bank.MPH_TO_FPS
    vb_r = vb_x * math.sin(phi_rad) + vb_y * math.cos(phi_rad)
    vb_t = vb_x * math.cos(phi_rad) - vb_y * math.sin(phi_rad)

    dv_r = vb_r - vB_r
    dv_z = vb_z - vB_z

    R_loc = bank.R_player(x_in, hitter)
    r_y_val = float(
        bank.r_y_of_x(
            x_in,
            hitter.x_cm,
            hitter.m_ball_oz,
            hitter.bat_weight_oz,
            hitter.I0_oz_in2,
        )
    )
    r_x_val = float(
        bank.r_x_of_x(
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
    alpha = float(hitter.alpha)

    cp, sp = math.cos(psi_rad), math.sin(psi_rad)
    vB_n = vB_r * cp + vB_z * sp
    vB_tn = -vB_r * sp + vB_z * cp
    vb_n = vb_r * cp + vb_z * sp
    vb_tn = -vb_r * sp + vb_z * cp
    d_n = dv_r * cp + dv_z * sp
    d_t = -dv_r * sp + dv_z * cp

    Vn_plus = float(vB_n + lam * d_n)

    denom_ft = alpha * (d_t - r_ball_ft * trans["omega_minus"])
    if abs(denom_ft) < DENOM_TOL:
        return _fail("bad_denom_Ft")

    Vt_plus = float(
        vB_tn
        + ((1.0 + e_x_in) * alpha * (d_t - r_ball_ft * trans["omega_minus"]))
        / ((1.0 + r_x_val) * (1.0 + alpha))
    )

    s_fps = math.hypot(Vn_plus, Vt_plus)
    if not np.isfinite(s_fps) or s_fps <= 0.0:
        return _fail("nonpositive_or_invalid_exit_speed")

    theta_rad = psi_rad + math.atan2(Vt_plus, Vn_plus)
    theta_star_deg = float(math.degrees(theta_rad))
    cos_delta = Vn_plus / s_fps if s_fps > 0 else float("nan")
    sin_delta = Vt_plus / s_fps if s_fps > 0 else float("nan")

    vt_over_vn_abs = np.inf if abs(Vn_plus) < DENOM_TOL else abs(Vt_plus) / abs(Vn_plus)

    psi_rel_legacy = math.atan2(vb_tn, vb_n)
    D_in = bank.D_of_psi_rel(R_loc, hitter.r_ball_in, psi_rel_legacy)

    e_x_raw = float(e_x_in)
    omega_plus = float("nan")
    if abs(denom_ft) >= DENOM_TOL:
        omega_plus = trans["omega_minus"] + (
            ((vB_tn - Vt_plus) - (D_in / hitter.r_ball_in) * Vn_plus) / (alpha * r_ball_ft)
        )

    regime_info = bank.classify_regime_from_psi_rel(
        abs(psi_rel_legacy), DEFAULT_MU, DEFAULT_G2, e_y_star
    )
    regime_label = regime_info["regime_label"]
    omega_minus = float(trans["omega_minus"])

    if not np.isfinite(omega_minus):
        return _fail("nonfinite_omega_minus")

    log_prior_ex = bank.log_prior_ex_from_regime(e_x_raw, regime_label)

    diag_flags: list[str] = []

    def _strict_checks() -> DecoderOutputs | None:
        if Vn_plus <= 0.0:
            return _fail("nonpositive_Vn_plus")
        if not (vt_over_vn_abs < 1.0):
            return _fail("vt_over_vn_ge_1")
        if d_n <= 0.0:
            return _fail("nonpositive_delta_n")
        if abs(D_in) > (R_loc + hitter.r_ball_in + 1e-8):
            return _fail("geometry_D_out_of_bounds")
        if regime_label == "gross-slip":
            return _fail("regime_gross_slip")
        if (not np.isfinite(e_x_raw)) or (not (0.0 <= e_x_raw <= 0.6)):
            return _fail("e_x_out_of_bounds")
        if not np.isfinite(log_prior_ex):
            return _fail("nonfinite_log_prior_ex")
        if abs(d_t - r_ball_ft * omega_minus) <= EPS_DT:
            return _fail("dt_minus_r_omega_near_zero")
        return None

    if admissibility == "strict":
        bad = _strict_checks()
        if bad is not None:
            return bad
    else:
        def _flag(name: str, cond: bool) -> None:
            if cond:
                diag_flags.append(name)

        _flag("nonpositive_Vn_plus", Vn_plus <= 0.0)
        _flag("vt_over_vn_ge_1", not (vt_over_vn_abs < 1.0))
        _flag("nonpositive_delta_n", d_n <= 0.0)
        _flag("geometry_D_out_of_bounds", abs(D_in) > (R_loc + hitter.r_ball_in + 1e-8))
        _flag("regime_gross_slip", regime_label == "gross-slip")
        _flag("e_x_out_of_bounds", (not np.isfinite(e_x_raw)) or (not (0.0 <= e_x_raw <= 0.6)))
        _flag("nonfinite_log_prior_ex", not np.isfinite(log_prior_ex))
        _flag("dt_minus_r_omega_near_zero", abs(d_t - r_ball_ft * omega_minus) <= EPS_DT)

    ev_mph = float(s_fps * bank.FPS_TO_MPH)
    omega_out = float(omega_plus) if np.isfinite(omega_plus) else None
    log_pe = float(log_prior_ex) if np.isfinite(log_prior_ex) else None
    flags_joined = ";".join(diag_flags) if diag_flags else None

    return DecoderOutputs(
        ev_mph=ev_mph,
        launch_angle_deg=theta_star_deg,
        spray_angle_deg=phi_star_deg,
        e_x=float(e_x_raw),
        omega_plus_rad_s=omega_out,
        admissible=True,
        failure_reason="accepted",
        V_n_plus_fps=float(Vn_plus),
        V_t_plus_fps=float(Vt_plus),
        cos_theta_minus_psi=float(cos_delta),
        sin_theta_minus_psi=float(sin_delta),
        log_prior_ex=log_pe,
        regime_label=regime_label,
        lambda_xy=float(lam),
        r_x_x=float(r_x_val),
        r_y_x=float(r_y_val),
        psi_rel_legacy_deg=float(math.degrees(psi_rel_legacy)),
        d_n=float(d_n),
        d_t=float(d_t),
        x_in=float(x_in),
        strict_diagnostic_flags=flags_joined,
    )


def decode_bip_from_sample(
    inp: DecoderInputs,
    *,
    admissibility: Literal["strict", "lenient"] = "strict",
) -> DecoderOutputs:
    """
    Deterministic decode one row. On hard failure, primary outputs are NaN and
    `failure_reason` is set (no soft fallback).

    **strict** — production inverse-style gates reject samples (default).
    **lenient** — forward generative benchmark: return (EV, LA, SA) whenever the
    forward kinematic closure yields a positive exit speed ``s_fps``; inverse/regime
    checks become ``strict_diagnostic_flags`` instead of hard rejection. True
    impossibilities (bad inputs, degenerate cos(θ−ψ), non-positive ``s_fps``, etc.)
    still fail.

    Raises:
        DecoderInputError: if ``inp`` violates the locked public contract.
    """
    validate_decoder_inputs(inp)
    if decoder_z_branch(inp.z) == "ex_analytic":
        return decode_bip_analytic_ex(inp, admissibility=admissibility)
    bank = get_bank_module()
    ic = get_inverse_collision_module()
    PSI_SUPPORT_DEG = ic.PSI_SUPPORT_DEG
    DENOM_TOL = ic.DENOM_TOL
    DEFAULT_MU = bank.DEFAULT_MU
    DEFAULT_G2 = bank.DEFAULT_G2_BASEBALL

    def _fail(reason: str) -> DecoderOutputs:
        return DecoderOutputs(
            ev_mph=float("nan"),
            launch_angle_deg=float("nan"),
            spray_angle_deg=float("nan"),
            e_x=None,
            omega_plus_rad_s=None,
            admissible=False,
            failure_reason=reason,
        )

    u, z, g, p = inp.u, inp.z, inp.g, inp.p
    try:
        hitter = hitter_from_p_dict(inp.batter_name, p)
        trans, pitch = build_transient_trans_and_pitch(u, g, bank)
    except Exception as ex:
        return _fail(f"input_parse:{type(ex).__name__}")

    x_in = float(z["x"])
    psi_deg = float(z["psi_deg"])
    e_y_star = float(z["e_y_star"])
    theta_deg = float(z["theta_deg"])

    diag_flags: list[str] = []
    if not (PSI_SUPPORT_DEG[0] <= psi_deg <= PSI_SUPPORT_DEG[1]):
        if admissibility == "strict":
            return _fail("psi_outside_support")
        diag_flags.append("psi_outside_support")

    if not np.isfinite(theta_deg) or not np.isfinite(psi_deg):
        return _fail("nonfinite_angles")

    phi_rad = math.radians(float(u["d_tilde"]))
    theta_rad = math.radians(theta_deg)
    psi_rad = math.radians(psi_deg)

    theta_star_deg = theta_deg
    phi_star_deg = float(u["d_tilde"])

    cos_delta = math.cos(theta_rad - psi_rad)
    sin_delta = math.sin(theta_rad - psi_rad)

    if abs(cos_delta) < DENOM_TOL:
        return _fail("degenerate_cos_theta_minus_psi")

    # Pitch / bat in horizontal radial frame (production)
    vB_x = float(pitch["vin_x"])
    vB_y = float(pitch["vin_y"])
    vB_z = float(pitch["vin_z"])
    vB_r = vB_x * math.sin(phi_rad) + vB_y * math.cos(phi_rad)
    vB_t = vB_x * math.cos(phi_rad) - vB_y * math.sin(phi_rad)

    try:
        bvec = ic.bat_vector_from_x(x_in, trans, hitter, bank)
    except Exception:
        return _fail("bat_vector_failure")

    vb_x = float(bvec["vx_bat_mph"]) * bank.MPH_TO_FPS
    vb_y = float(bvec["vy_bat_mph"]) * bank.MPH_TO_FPS
    vb_z = float(bvec["vz_bat_mph"]) * bank.MPH_TO_FPS
    vb_r = vb_x * math.sin(phi_rad) + vb_y * math.cos(phi_rad)
    vb_t = vb_x * math.cos(phi_rad) - vb_y * math.sin(phi_rad)

    dv_r = vb_r - vB_r
    dv_z = vb_z - vB_z

    R_loc = bank.R_player(x_in, hitter)
    r_y_val = float(
        bank.r_y_of_x(
            x_in,
            hitter.x_cm,
            hitter.m_ball_oz,
            hitter.bat_weight_oz,
            hitter.I0_oz_in2,
        )
    )
    r_x_val = float(
        bank.r_x_of_x(
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

    cp, sp = math.cos(psi_rad), math.sin(psi_rad)
    vB_n = vB_r * cp + vB_z * sp
    vB_tn = -vB_r * sp + vB_z * cp
    vb_n = vb_r * cp + vb_z * sp
    vb_tn = -vb_r * sp + vb_z * cp
    d_n = dv_r * cp + dv_z * sp
    d_t = -dv_r * sp + dv_z * cp

    # F_n = 0 closure at fixed psi (same as bank.F_n_scalar_from_ev with e_y_star for lambda)
    Vn_plus = float(vB_n + lam * d_n)

    s_fps = Vn_plus / cos_delta
    if not np.isfinite(s_fps) or s_fps <= 0.0:
        return _fail("nonpositive_or_invalid_exit_speed")

    Vt_plus = float(s_fps * sin_delta)

    vt_over_vn_abs = np.inf if abs(Vn_plus) < DENOM_TOL else abs(Vt_plus) / abs(Vn_plus)

    psi_rel_legacy = math.atan2(vb_tn, vb_n)
    D_in = bank.D_of_psi_rel(R_loc, hitter.r_ball_in, psi_rel_legacy)
    denom_ft = float(hitter.alpha * (d_t - r_ball_ft * trans["omega_minus"]))

    e_x_raw = float("nan")
    omega_plus = float("nan")
    if abs(denom_ft) >= DENOM_TOL:
        e_x_raw = ((Vt_plus - vB_tn) * (1.0 + r_x_val) * (1.0 + hitter.alpha)) / denom_ft - 1.0
        omega_plus = trans["omega_minus"] + (
            ((vB_tn - Vt_plus) - (D_in / hitter.r_ball_in) * Vn_plus) / (hitter.alpha * r_ball_ft)
        )

    regime_info = bank.classify_regime_from_psi_rel(
        abs(psi_rel_legacy), DEFAULT_MU, DEFAULT_G2, e_y_star
    )
    regime_label = regime_info["regime_label"]
    omega_minus = float(trans["omega_minus"])

    if not np.isfinite(omega_minus):
        return _fail("nonfinite_omega_minus")

    log_prior_ex = bank.log_prior_ex_from_regime(e_x_raw, regime_label)

    if admissibility == "strict":
        # --- Hard admissibility (production branch order; see solve_exact_root_state) ---
        if Vn_plus <= 0.0:
            return _fail("nonpositive_Vn_plus")
        if not (vt_over_vn_abs < 1.0):
            return _fail("vt_over_vn_ge_1")
        if d_n <= 0.0:
            return _fail("nonpositive_delta_n")
        if abs(D_in) > (R_loc + hitter.r_ball_in + 1e-8):
            return _fail("geometry_D_out_of_bounds")
        if abs(denom_ft) < DENOM_TOL:
            return _fail("bad_denom_Ft")
        if regime_label == "gross-slip":
            return _fail("regime_gross_slip")
        if (not np.isfinite(e_x_raw)) or (not (0.0 <= e_x_raw <= 0.6)):
            return _fail("e_x_out_of_bounds")
        if not np.isfinite(log_prior_ex):
            return _fail("nonfinite_log_prior_ex")

        ev_mph = float(s_fps * bank.FPS_TO_MPH)

        return DecoderOutputs(
            ev_mph=ev_mph,
            launch_angle_deg=float(theta_star_deg),
            spray_angle_deg=float(phi_star_deg),
            e_x=float(e_x_raw),
            omega_plus_rad_s=float(omega_plus),
            admissible=True,
            failure_reason="accepted",
            V_n_plus_fps=float(Vn_plus),
            V_t_plus_fps=float(Vt_plus),
            cos_theta_minus_psi=float(cos_delta),
            sin_theta_minus_psi=float(sin_delta),
            log_prior_ex=float(log_prior_ex),
            regime_label=regime_label,
            lambda_xy=float(lam),
            r_x_x=float(r_x_val),
            r_y_x=float(r_y_val),
            psi_rel_legacy_deg=float(math.degrees(psi_rel_legacy)),
            d_n=float(d_n),
            d_t=float(d_t),
            x_in=float(x_in),
            strict_diagnostic_flags=None,
        )

    # lenient: same (EV, LA, SA) whenever s_fps > 0; inverse gates -> strict_diagnostic_flags
    def _flag(name: str, cond: bool) -> None:
        if cond:
            diag_flags.append(name)

    _flag("nonpositive_Vn_plus", Vn_plus <= 0.0)
    _flag("vt_over_vn_ge_1", not (vt_over_vn_abs < 1.0))
    _flag("nonpositive_delta_n", d_n <= 0.0)
    _flag("geometry_D_out_of_bounds", abs(D_in) > (R_loc + hitter.r_ball_in + 1e-8))
    _flag("bad_denom_Ft", abs(denom_ft) < DENOM_TOL)
    _flag("regime_gross_slip", regime_label == "gross-slip")
    _flag(
        "e_x_out_of_bounds",
        (not np.isfinite(e_x_raw)) or (not (0.0 <= e_x_raw <= 0.6)),
    )
    _flag("nonfinite_log_prior_ex", not np.isfinite(log_prior_ex))

    ev_mph = float(s_fps * bank.FPS_TO_MPH)
    e_x_out = float(e_x_raw) if np.isfinite(e_x_raw) else None
    omega_out = float(omega_plus) if np.isfinite(omega_plus) else None
    log_pe = float(log_prior_ex) if np.isfinite(log_prior_ex) else None
    flags_joined = ";".join(diag_flags) if diag_flags else None

    return DecoderOutputs(
        ev_mph=ev_mph,
        launch_angle_deg=float(theta_star_deg),
        spray_angle_deg=float(phi_star_deg),
        e_x=e_x_out,
        omega_plus_rad_s=omega_out,
        admissible=True,
        failure_reason="accepted",
        V_n_plus_fps=float(Vn_plus),
        V_t_plus_fps=float(Vt_plus),
        cos_theta_minus_psi=float(cos_delta),
        sin_theta_minus_psi=float(sin_delta),
        log_prior_ex=log_pe,
        regime_label=regime_label,
        lambda_xy=float(lam),
        r_x_x=float(r_x_val),
        r_y_x=float(r_y_val),
        psi_rel_legacy_deg=float(math.degrees(psi_rel_legacy)),
        d_n=float(d_n),
        d_t=float(d_t),
        x_in=float(x_in),
        strict_diagnostic_flags=flags_joined,
    )


def decode_bip_batch(
    samples: list[DecoderInputs],
    *,
    admissibility: Literal["strict", "lenient"] = "strict",
) -> pd.DataFrame:
    """Vectorized over rows; each row independent."""
    rows = []
    for inp in samples:
        out = decode_bip_from_sample(inp, admissibility=admissibility)
        d = asdict(out)
        d["event_id"] = inp.event_id
        d["batter_name"] = inp.batter_name
        d["EV"] = d.pop("ev_mph")
        d["LA"] = d.pop("launch_angle_deg")
        d["SA"] = d.pop("spray_angle_deg")
        rows.append(d)
    return pd.DataFrame(rows)


def decode_to_dict(inp: DecoderInputs) -> dict[str, Any]:
    """Flat dict for logging / tests."""
    return asdict(decode_bip_from_sample(inp))


class ExactRootPhysicsDecoder:
    """`PhysicsDecoder` protocol implementation via `decode_bip_from_sample`."""

    def decode(self, inp: DecoderInputs) -> DecoderOutputs:
        return decode_bip_from_sample(inp)
