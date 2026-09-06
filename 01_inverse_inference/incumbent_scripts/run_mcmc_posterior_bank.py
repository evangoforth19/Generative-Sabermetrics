
# =========================================================
# FULL POSTERIOR ARTIFACT RUNNER
# =========================================================

import argparse
import gzip
import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

# Keep defaults above function signatures that reference them.
DEFAULT_DENOM_TOL = 1e-5
DEFAULT_ROOT_TOL = 1e-5
DEFAULT_MU = 0.5
DEFAULT_G2_BASEBALL = 4.75
DEFAULT_SIGMA_N = 0.1
DEFAULT_SIGMA_T = 0.1
DEFAULT_SIGMA_W = 0.1


def display(x):
    print(x)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_table(df, path_no_ext):
    path_no_ext = Path(path_no_ext)
    try:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    except Exception:
        out = path_no_ext.with_suffix(".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
        return out


def _write_json(obj, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _find_interval_id(x_val, intervals):
    for k, (a, b) in enumerate(intervals):
        if (x_val >= a - 1e-12) and (x_val <= b + 1e-12):
            return int(k)
    return -1


def _interval_mass_probs(manifold_map):
    if manifold_map is None or manifold_map.get("map") is None:
        return {}
    x = manifold_map["x"]
    valid = manifold_map["admissible_mask"]
    if len(x) == 0 or not np.any(valid):
        return {}
    x_valid = x[valid]
    iid_valid = manifold_map["interval_id"][valid]
    lt_valid = manifold_map["map"]["log_target"][valid]
    cell_w = _midpoint_cell_widths(x_valid)
    w = _safe_exp_normalized(lt_valid) * cell_w
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return {}
    p = w / w_sum
    out = {}
    for iid in np.unique(iid_valid):
        out[int(iid)] = float(p[iid_valid == iid].sum())
    return out


def _weighted_map_moments(manifold_map):
    if manifold_map is None or manifold_map.get("map") is None:
        return {}
    x = manifold_map["x"]
    valid = manifold_map["admissible_mask"]
    if len(x) == 0 or not np.any(valid):
        return {}

    mm = manifold_map["map"]
    x_valid = x[valid]
    lt_valid = mm["log_target"][valid]
    cell_w = _midpoint_cell_widths(x_valid)
    w = _safe_exp_normalized(lt_valid) * cell_w
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return {}
    p = w / w_sum

    def _mom(arr):
        arr = np.asarray(arr)[valid]
        mu = float(np.sum(p * arr))
        var = float(np.sum(p * (arr - mu) ** 2))
        return mu, math.sqrt(max(var, 0.0))

    x_mu, x_sd = _mom(mm["x"])
    psi_mu, psi_sd = _mom(np.rad2deg(mm["psi_abs"]))
    ex_mu, ex_sd = _mom(mm["e_x"])
    branch_probs = _interval_mass_probs(manifold_map)
    branch_entropy = float(-sum(prob * np.log(prob) for prob in branch_probs.values() if prob > 0.0)) if branch_probs else np.nan

    return {
        "map_weighted_x_mean": x_mu,
        "map_weighted_x_sd": x_sd,
        "map_weighted_psi_abs_deg_mean": psi_mu,
        "map_weighted_psi_abs_deg_sd": psi_sd,
        "map_weighted_e_x_mean": ex_mu,
        "map_weighted_e_x_sd": ex_sd,
        "map_branch_entropy": branch_entropy,
    }


def _measurement_correction_row(raw_star, raw_obs, sensor_sigmas):
    deltas = {}
    zsq = []
    abs_deltas = []
    for k, obs_val in raw_obs.items():
        if k == "active_spin_ratio":
            continue
        if k not in raw_star:
            continue
        d = float(raw_star[k] - obs_val)
        deltas[f"delta_{k}"] = d
        abs_deltas.append(abs(d))
        sd = float(sensor_sigmas.get(k, np.nan))
        if np.isfinite(sd) and sd > 0.0:
            zsq.append((d / sd) ** 2)
            deltas[f"delta_z_{k}"] = float(d / sd)

    deltas["measurement_correction_l2"] = float(np.sqrt(np.sum(np.square(abs_deltas)))) if abs_deltas else np.nan
    deltas["measurement_correction_std_l2"] = float(np.sqrt(np.sum(zsq))) if zsq else np.nan
    deltas["measurement_correction_max_abs"] = float(np.max(abs_deltas)) if abs_deltas else np.nan
    return deltas


def _gaussian_entropy_from_cov(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        return np.nan
    if not np.all(np.isfinite(cov)):
        return np.nan
    jitter = 1e-10 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov + jitter)
    if sign <= 0:
        return np.nan
    k = cov.shape[0]
    return float(0.5 * (k * np.log(2.0 * np.pi * np.e) + logdet))


def _series_mode_prob(vals):
    vals = pd.Series(vals)
    probs = vals.value_counts(normalize=True, dropna=False)
    return {str(k): float(v) for k, v in probs.items()}


def _hdi(vals, prob=0.90):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    try:
        lo, hi = az.hdi(vals, hdi_prob=prob)
        return float(lo), float(hi)
    except Exception:
        qlo = (1.0 - prob) / 2.0
        qhi = 1.0 - qlo
        lo, hi = np.quantile(vals, [qlo, qhi])
        return float(lo), float(hi)


def _summary_for_array(vals, prefix):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q50": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_hdi90_lo": np.nan,
            f"{prefix}_hdi90_hi": np.nan,
            f"{prefix}_width90": np.nan,
        }
    q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
    hdi_lo, hdi_hi = _hdi(vals, 0.90)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        f"{prefix}_q05": float(q05),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q95": float(q95),
        f"{prefix}_hdi90_lo": float(hdi_lo),
        f"{prefix}_hdi90_hi": float(hdi_hi),
        f"{prefix}_width90": float(q95 - q05),
    }


def _cov_corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 2:
        return np.nan, np.nan
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a, ddof=1) > 0 and np.std(b, ddof=1) > 0 else np.nan
    return cov, corr


def summarize_event_posterior(event_id, event_row, hitter, posterior_df, init_diag):
    summary = {
        "event_id": int(event_id),
        "orig_index": int(event_id),
        "game_date": str(event_row.get("game_date", "")),
        "batter_name": hitter.name,
        "pitch_type": str(event_row.get("pitch_type", "")),
        "stand": str(event_row.get("stand", "")),
        "p_throws": str(event_row.get("p_throws", "")),
        "balls": int(event_row.get("balls", -1)) if pd.notna(event_row.get("balls", np.nan)) else np.nan,
        "strikes": int(event_row.get("strikes", -1)) if pd.notna(event_row.get("strikes", np.nan)) else np.nan,
        "count_str": str(event_row.get("count_str", "")),
        "z_count": float(event_row.get("z_count", np.nan)),
        "plate_x": float(event_row.get("plate_x", np.nan)),
        "plate_z": float(event_row.get("plate_z", np.nan)),
        "launch_speed_obs_mph": float(event_row.get("launch_speed", np.nan)),
        "launch_angle_obs_deg": float(event_row.get("launch_angle", np.nan)),
        "bat_speed_obs_mph": float(event_row.get("bat_speed", np.nan)),
        "x_cm_fixed_in": float(hitter.x_cm),
        "r_g_fixed_in": float(hitter.r_g),
        "x_cm_posterior_available": False,
        "r_g_posterior_available": False,
        "n_draws": int(len(posterior_df)),
        "init_try": int(init_diag.get("init_try", np.nan)),
        "init_failure_top_reason_before_success": _series_top_reason(init_diag.get("init_reason_counter", {})),
        "init_reason_counter_json": json.dumps(init_diag.get("init_reason_counter", {}), default=_json_default),
    }

    cont_cols = [
        "z_x_in",
        "z_theta_deg",
        "z_phi_deg",
        "u_bat_speed_mph",
        "u_attack_angle_deg",
        "u_attack_direction_deg",
        "latent_psi_abs_deg",
        "latent_psi_rel_deg",
        "latent_e_x",
        "latent_e_y",
        "latent_omega_plus",
        "latent_omega_minus",
        "diag_R_prof",
        "diag_Fn_res",
        "diag_Ft_res",
        "diag_Fw_res",
        "measurement_correction_l2",
        "measurement_correction_std_l2",
        "map_weighted_x_mean",
        "map_weighted_x_sd",
        "map_weighted_psi_abs_deg_mean",
        "map_weighted_e_x_mean",
    ]
    for col in cont_cols:
        if col in posterior_df.columns:
            summary.update(_summary_for_array(posterior_df[col].values, col))

    cov_x_psi, corr_x_psi = _cov_corr_safe(posterior_df["z_x_in"].values, posterior_df["latent_psi_abs_deg"].values)
    cov_x_ex, corr_x_ex = _cov_corr_safe(posterior_df["z_x_in"].values, posterior_df["latent_e_x"].values)
    summary["cov_x_psi_abs_deg"] = cov_x_psi
    summary["corr_x_psi_abs_deg"] = corr_x_psi
    summary["cov_x_e_x"] = cov_x_ex
    summary["corr_x_e_x"] = corr_x_ex

    summary["mstar_accept_rate"] = float(posterior_df["mstar_accepted"].mean()) if "mstar_accepted" in posterior_df else np.nan
    summary["mstar_valid_prop_rate"] = float(posterior_df["mstar_valid_prop"].mean()) if "mstar_valid_prop" in posterior_df else np.nan
    summary["direct_x_resample_success_rate"] = float(posterior_df["direct_x_resample_success"].mean()) if "direct_x_resample_success" in posterior_df else np.nan
    summary["map_rebuilt_rate"] = float(posterior_df["map_rebuilt"].mean()) if "map_rebuilt" in posterior_df else np.nan

    summary["map_interval_count_mean"] = float(posterior_df["map_interval_count"].mean()) if "map_interval_count" in posterior_df else np.nan
    summary["map_total_width_mean"] = float(posterior_df["map_total_width"].mean()) if "map_total_width" in posterior_df else np.nan
    summary["map_branch_entropy_mean"] = float(posterior_df["map_branch_entropy"].mean()) if "map_branch_entropy" in posterior_df else np.nan

    # Regime and branch probabilities
    regime_probs = _series_mode_prob(posterior_df["regime_label"]) if "regime_label" in posterior_df else {}
    branch_probs = _series_mode_prob(posterior_df["branch_label"]) if "branch_label" in posterior_df else {}
    summary["regime_prob_stick_slip"] = float(regime_probs.get("stick-slip", 0.0))
    summary["regime_prob_slip_stick_slip"] = float(regime_probs.get("slip-stick-slip", 0.0))
    summary["regime_prob_gross_slip"] = float(regime_probs.get("gross-slip", 0.0))
    summary["branch_entropy_discrete"] = float(-sum(p * np.log(p) for p in branch_probs.values() if p > 0.0)) if branch_probs else np.nan

    # Gaussian plug-in entropy / uncertainty volume
    entropy_vars = ["z_x_in", "latent_psi_abs_deg", "latent_e_x", "latent_omega_plus"]
    existing_entropy_vars = [c for c in entropy_vars if c in posterior_df.columns]
    if existing_entropy_vars:
        X = posterior_df[existing_entropy_vars].astype(float).to_numpy()
        keep = np.isfinite(X).all(axis=1)
        X = X[keep]
        if len(X) >= 2:
            cov = np.cov(X, rowvar=False, ddof=1)
            summary["posterior_entropy_gaussian_plugin"] = _gaussian_entropy_from_cov(cov)
            try:
                summary["uncertainty_volume_sqrt_det_cov"] = float(np.sqrt(max(np.linalg.det(cov), 0.0)))
            except Exception:
                summary["uncertainty_volume_sqrt_det_cov"] = np.nan
        else:
            summary["posterior_entropy_gaussian_plugin"] = np.nan
            summary["uncertainty_volume_sqrt_det_cov"] = np.nan
    else:
        summary["posterior_entropy_gaussian_plugin"] = np.nan
        summary["uncertainty_volume_sqrt_det_cov"] = np.nan

    return summary, regime_probs, branch_probs


def build_covariance_tables(event_id, posterior_df):
    cov_vars = [
        "z_x_in",
        "z_theta_deg",
        "z_phi_deg",
        "u_bat_speed_mph",
        "u_attack_angle_deg",
        "u_attack_direction_deg",
        "latent_psi_abs_deg",
        "latent_psi_rel_deg",
        "latent_e_x",
        "latent_e_y",
        "latent_omega_plus",
        "latent_omega_minus",
    ]
    cov_vars = [c for c in cov_vars if c in posterior_df.columns]
    if len(cov_vars) == 0:
        return pd.DataFrame(), pd.DataFrame()
    X = posterior_df[cov_vars].astype(float)
    keep = np.isfinite(X).all(axis=1)
    X = X.loc[keep]
    if len(X) < 2:
        return pd.DataFrame(), pd.DataFrame()

    cov = X.cov()
    corr = X.corr()

    cov_rows = []
    corr_rows = []
    for i in cov.index:
        for j in cov.columns:
            cov_rows.append({"event_id": int(event_id), "var_i": i, "var_j": j, "cov": float(cov.loc[i, j])})
            corr_rows.append({"event_id": int(event_id), "var_i": i, "var_j": j, "corr": float(corr.loc[i, j])})
    return pd.DataFrame(cov_rows), pd.DataFrame(corr_rows)


def run_single_event_fast_full_chain(
    event_row,
    hitter,
    event_id,
    n_iter=800,
    burn=200,
    thin=2,
    seed=123,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=0.1,
    sigma_t=0.1,
    sigma_w=0.1,
    coarse_n=61,
    fine_points_per_interval=80,
):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    rng = np.random.default_rng(seed)

    init = initialize_fast_chain_with_diagnostics(
        event_row=event_row,
        hitter=hitter,
        rng=rng,
        sensor_sigmas=sensor_sigmas,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
        coarse_n=coarse_n,
        fine_points_per_interval=fine_points_per_interval,
    )

    if not init["success"]:
        return {"success": False, "event_id": int(event_id), "init_diagnostics": init}

    ev_obs_raw = init["ev_obs_raw"]
    current_raw_star = init["raw_star"]
    current_ev_star = init["ev_star"]
    current_x = init["x_in"]
    current_state = init["state"]
    current_map = init["manifold_map"]

    draws = []
    valid_mstar_prop = 0
    accept_mstar = 0

    for it in range(n_iter):
        map_rebuilt = 0
        mstar_accepted = 0
        mstar_valid_prop = 0

        prop_raw_star = sample_measurement_star(ev_obs_raw, rng, sensor_sigmas=sensor_sigmas)
        prop_ev_star = rebuild_observed_event(prop_raw_star)

        prop_ok = np.all(np.isfinite([
            prop_ev_star["phi"], prop_ev_star["theta"], prop_ev_star["s_obs"],
            prop_ev_star["omega_minus"],
            prop_ev_star["vB_r_m"], prop_ev_star["vB_z_m"],
            prop_ev_star["vb_r_m"], prop_ev_star["vb_z_m"],
        ]))

        if prop_ok:
            prop_state_at_current_x = evaluate_x_state(
                x_in=current_x,
                hitter=hitter,
                ev_star=prop_ev_star,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
            )
            if prop_state_at_current_x is not None:
                mstar_valid_prop = 1
                valid_mstar_prop += 1
                log_alpha = float(prop_state_at_current_x["log_target"] - current_state["log_target"])
                if np.log(rng.uniform()) < log_alpha:
                    current_raw_star = prop_raw_star
                    current_ev_star = prop_ev_star
                    current_state = prop_state_at_current_x
                    current_map = build_fast_manifold_map(
                        hitter=hitter,
                        ev_star=current_ev_star,
                        psi_abs_min_deg=psi_abs_min_deg,
                        psi_abs_max_deg=psi_abs_max_deg,
                        mu=mu,
                        g2=g2,
                        sigma_n=sigma_n,
                        sigma_t=sigma_t,
                        sigma_w=sigma_w,
                        coarse_n=coarse_n,
                        fine_points_per_interval=fine_points_per_interval,
                    )
                    map_rebuilt = 1
                    mstar_accepted = 1
                    accept_mstar += 1

        x_prop, st_prop, x_info = sample_x_from_fast_manifold_map(
            manifold_map=current_map,
            hitter=hitter,
            ev_star=current_ev_star,
            rng=rng,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )

        if st_prop is not None:
            current_x = float(x_prop)
            current_state = st_prop

        if (it >= burn) and (((it - burn) % thin) == 0):
            branch_probs = _interval_mass_probs(current_map)
            branch_label = _find_interval_id(current_x, current_map["intervals"])
            map_moments = _weighted_map_moments(current_map)
            corr_row = _measurement_correction_row(current_raw_star, ev_obs_raw, sensor_sigmas)

            draw = {
                "event_id": int(event_id),
                "draw_idx": int((it - burn) // thin),
                "iter": int(it),

                # predictive latent z
                "z_x_in": float(current_x),
                "z_theta_deg": float(np.rad2deg(current_ev_star["theta_abs"])),
                "z_phi_deg": float(np.rad2deg(current_ev_star["phi"])),

                # upstream u from denoised sensor layer
                "u_bat_speed_mph": float(current_raw_star["bat_speed_mph"]),
                "u_attack_angle_deg": float(current_raw_star["attack_angle_deg"]),
                "u_attack_direction_deg": float(current_raw_star["attack_direction_deg"]),

                # full inverse latent state
                "latent_psi_abs_deg": float(np.rad2deg(current_state["psi_abs"])),
                "latent_psi_rel_deg": float(np.rad2deg(current_state["psi_rel"])),
                "latent_e_x": float(current_state["e_x"]),
                "latent_e_y": float(current_state["e_y"]),
                "latent_omega_plus": float(current_state["omega_plus"]),
                "latent_omega_minus": float(current_state["omega_minus"]),
                "latent_Vn_plus_obs": float(current_state["Vn_plus_obs"]),
                "latent_Vt_plus_obs": float(current_state["Vt_plus_obs"]),
                "latent_vt_over_vn_abs": float(current_state["vt_over_vn_abs"]),
                "latent_D_in": float(current_state["D_in"]),
                "latent_R_in": float(current_state["R_in"]),
                "latent_r_x": float(current_state["r_x"]),
                "latent_r_y": float(current_state["r_y"]),

                # branch / regime
                "branch_label": int(branch_label),
                "branch_prob_0": float(branch_probs.get(0, 0.0)),
                "branch_prob_1": float(branch_probs.get(1, 0.0)),
                "branch_probs_json": json.dumps(branch_probs, default=_json_default),
                "regime_label": str(current_state["regime_label"]),
                "regime_lower": float(current_state["regime_lower"]),
                "regime_upper": float(current_state["regime_upper"]),
                "tan_psi_regime": float(current_state["tan_psi_regime"]),

                # diagnostics
                "diag_R_prof": float(current_state["R_prof"]),
                "diag_Fn_res": float(current_state["Fn_res"]),
                "diag_Ft_res": float(current_state["Ft_res"]),
                "diag_Fw_res": float(current_state["Fw_res"]),
                "diag_log_target": float(current_state["log_target"]),
                "diag_log_prior_ex": float(current_state["log_prior_ex"]),
                "mstar_accepted": int(mstar_accepted),
                "mstar_valid_prop": int(mstar_valid_prop),
                "direct_x_resample_success": int(x_info["direct_x_resample_success"]),
                "map_rebuilt": int(map_rebuilt),
                "map_interval_count": int(x_info["map_interval_count"]),
                "map_total_width": float(x_info["map_total_width"]),
                "map_total_points": int(x_info["map_total_points"]),
                "init_coarse_n_admissible": int(init["coarse_n_admissible"]),
                "init_map_interval_count": int(init["map_interval_count"]),
            }
            draw.update(map_moments)
            draw.update(corr_row)
            draws.append(draw)

    posterior_df = pd.DataFrame(draws)
    if posterior_df.empty:
        return {
            "success": False,
            "event_id": int(event_id),
            "init_diagnostics": init,
            "failure_reason": "empty_posterior_after_burn",
        }

    summary, regime_probs, branch_probs = summarize_event_posterior(
        event_id=event_id,
        event_row=event_row,
        hitter=hitter,
        posterior_df=posterior_df,
        init_diag=init,
    )
    cov_df, corr_df = build_covariance_tables(event_id, posterior_df)

    branch_rows = []
    if "branch_label" in posterior_df.columns:
        post_branch_probs = posterior_df["branch_label"].value_counts(normalize=True).to_dict()
        for branch, prob in sorted(post_branch_probs.items()):
            sub = posterior_df.loc[posterior_df["branch_label"] == branch]
            branch_rows.append({
                "event_id": int(event_id),
                "branch_label": int(branch),
                "posterior_branch_prob": float(prob),
                "avg_map_branch_prob": float(sub[f"branch_prob_{branch}"].mean()) if f"branch_prob_{branch}" in sub else np.nan,
                "mean_x_in": float(sub["z_x_in"].mean()) if len(sub) else np.nan,
                "q05_x_in": float(sub["z_x_in"].quantile(0.05)) if len(sub) else np.nan,
                "q95_x_in": float(sub["z_x_in"].quantile(0.95)) if len(sub) else np.nan,
            })
    branch_df = pd.DataFrame(branch_rows)

    regime_df = pd.DataFrame([
        {"event_id": int(event_id), "regime_label": str(k), "posterior_prob": float(v)}
        for k, v in regime_probs.items()
    ])

    return {
        "success": True,
        "event_id": int(event_id),
        "posterior_df": posterior_df,
        "summary": summary,
        "cov_df": cov_df,
        "corr_df": corr_df,
        "branch_df": branch_df,
        "regime_df": regime_df,
        "init_diagnostics": init,
    }


def run_success_target_backfill(
    bip_model,
    hitters,
    target_successes=1000,
    seed=123,
    n_iter=800,
    burn=200,
    thin=2,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=0.1,
    sigma_t=0.1,
    sigma_w=0.1,
    coarse_n=61,
    fine_points_per_interval=80,
):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    hitter_lookup = {h.name: h for h in hitters}

    work = bip_model.copy()
    work["game_date_dt"] = pd.to_datetime(work["game_date"], errors="coerce")
    work["orig_index"] = work.index.astype(int)
    sort_cols = ["game_date_dt", "orig_index"]
    asc = [False, False]
    if "pitch_number" in work.columns:
        sort_cols = ["game_date_dt", "pitch_number", "orig_index"]
        asc = [False, False, False]
    work = work.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

    results = []
    summary_rows = []
    failure_rows = []
    cov_rows = []
    corr_rows = []
    branch_rows = []
    regime_rows = []
    draw_frames = []

    rng_master = np.random.default_rng(seed)
    success_count = 0
    processed = 0

    iterator = tqdm(work.itertuples(index=False), total=len(work), desc="posterior bank", unit="event")

    for row in iterator:
        if success_count >= target_successes:
            break
        processed += 1
        event_id = int(row.orig_index)
        row_series = bip_model.loc[event_id]
        hitter_name = str(row_series["batter_name"]).lower()

        if hitter_name not in hitter_lookup:
            failure_rows.append({
                "event_id": event_id,
                "batter_name": hitter_name,
                "failure_top_reason": "missing_hitter_object",
                "game_date": str(row_series.get("game_date", "")),
            })
            iterator.set_postfix(success=success_count, processed=processed, last="missing_hitter")
            continue

        hitter = hitter_lookup[hitter_name]
        event_seed = int(rng_master.integers(1, 10_000_000))

        out = run_single_event_fast_full_chain(
            event_row=row_series,
            hitter=hitter,
            event_id=event_id,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            seed=event_seed,
            sensor_sigmas=sensor_sigmas,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
            coarse_n=coarse_n,
            fine_points_per_interval=fine_points_per_interval,
        )

        if out["success"]:
            success_count += 1
            draw_frames.append(out["posterior_df"])
            summary_rows.append(out["summary"])
            if not out["cov_df"].empty:
                cov_rows.append(out["cov_df"])
            if not out["corr_df"].empty:
                corr_rows.append(out["corr_df"])
            if not out["branch_df"].empty:
                branch_rows.append(out["branch_df"])
            if not out["regime_df"].empty:
                regime_rows.append(out["regime_df"])
            iterator.set_postfix(success=success_count, processed=processed, last="success")
        else:
            init_diag = out.get("init_diagnostics", {})
            failure_rows.append({
                "event_id": event_id,
                "batter_name": hitter.name,
                "game_date": str(row_series.get("game_date", "")),
                "failure_top_reason": init_diag.get("failure_top_reason", out.get("failure_reason", "unknown")),
                "init_try": int(init_diag.get("init_try", np.nan)) if init_diag else np.nan,
                "init_reason_counter_json": json.dumps(init_diag.get("init_reason_counter", {}), default=_json_default) if init_diag else "{}",
            })
            iterator.set_postfix(success=success_count, processed=processed, last="fail")

    combined_draws = pd.concat(draw_frames, ignore_index=True) if draw_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    failure_df = pd.DataFrame(failure_rows)
    cov_df = pd.concat(cov_rows, ignore_index=True) if cov_rows else pd.DataFrame()
    corr_df = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()
    branch_df = pd.concat(branch_rows, ignore_index=True) if branch_rows else pd.DataFrame()
    regime_df = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()

    return {
        "draws": combined_draws,
        "summary": summary_df,
        "failures": failure_df,
        "cov": cov_df,
        "corr": corr_df,
        "branch": branch_df,
        "regime": regime_df,
        "success_count": int(success_count),
        "processed_count": int(processed),
        "exhausted_dataset": bool(success_count < target_successes),
    }


def build_hitter_constants_table(hitters):
    rows = []
    for h in hitters:
        rows.append({
            "batter_name": h.name,
            "x_cm_fixed_in": float(h.x_cm),
            "r_g_fixed_in": float(h.r_g),
            "bat_length_in": float(h.bat_length),
            "bat_weight_oz": float(h.bat_weight_oz),
            "I0_oz_in2": float(h.I0_oz_in2),
            "Iz_oz_in2": float(h.Iz_oz_in2),
            "x_cm_posterior_available": False,
            "r_g_posterior_available": False,
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Run fast manifold MCMC and build posterior training artifacts.")
    p.add_argument("--data-pickle", type=str, default="batter_data_2020_2025.pkl", help="Path to batter_data pickle.")
    p.add_argument("--output-dir", type=str, default="./posterior_bank_outputs", help="Directory for outputs.")
    p.add_argument("--target-successes", type=int, default=1000, help="Number of successful event posteriors to keep.")
    p.add_argument("--seed", type=int, default=123, help="Master RNG seed.")
    p.add_argument("--n-iter", type=int, default=800, help="Total MCMC iterations per event.")
    p.add_argument("--burn", type=int, default=200, help="Burn-in iterations.")
    p.add_argument("--thin", type=int, default=2, help="Thinning interval.")
    p.add_argument("--psi-abs-min-deg", type=float, default=-85.0)
    p.add_argument("--psi-abs-max-deg", type=float, default=85.0)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--g2", type=float, default=4.75)
    p.add_argument("--sigma-n", type=float, default=0.1)
    p.add_argument("--sigma-t", type=float, default=0.1)
    p.add_argument("--sigma-w", type=float, default=0.1)
    p.add_argument("--coarse-n", type=int, default=61)
    p.add_argument("--fine-points-per-interval", type=int, default=80)
    p.add_argument("--quiet-dataset-build", action="store_true", help="Suppress dataset build sanity prints.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    batter_data_ext = pd.read_pickle(args.data_pickle)
    hitters, hitter_meta, bip_sub, bip_model, hitter_order = build_hitters_and_bip_model(
        batter_data_ext=batter_data_ext,
        verbose=(not args.quiet_dataset_build),
    )

    sensor_sigmas = default_sensor_sigmas()
    hitter_constants_df = build_hitter_constants_table(hitters)

    results = run_success_target_backfill(
        bip_model=bip_model,
        hitters=hitters,
        target_successes=args.target_successes,
        seed=args.seed,
        n_iter=args.n_iter,
        burn=args.burn,
        thin=args.thin,
        sensor_sigmas=sensor_sigmas,
        psi_abs_min_deg=args.psi_abs_min_deg,
        psi_abs_max_deg=args.psi_abs_max_deg,
        mu=args.mu,
        g2=args.g2,
        sigma_n=args.sigma_n,
        sigma_t=args.sigma_t,
        sigma_w=args.sigma_w,
        coarse_n=args.coarse_n,
        fine_points_per_interval=args.fine_points_per_interval,
    )

    event_metadata_cols = [
        "game_date", "batter_name", "pitcher", "pitch_number",
        "pitch_type", "stand", "p_throws", "balls", "strikes",
        "count_str", "z_count", "plate_x", "plate_z",
        "launch_speed", "launch_angle", "bat_speed",
        "vx0", "vy0", "vz0", "ax", "ay", "az",
        "release_pos_y", "release_spin_rate", "spin_axis",
        "h_idx",
    ]
    event_metadata_df = bip_model.loc[
        bip_model.index.isin(results["summary"]["event_id"]) if not results["summary"].empty else [],
        event_metadata_cols
    ].copy() if not results["summary"].empty else pd.DataFrame(columns=event_metadata_cols)
    if not event_metadata_df.empty:
        event_metadata_df = event_metadata_df.reset_index().rename(columns={"index": "event_id"})

    written = {}
    written["posterior_draws"] = str(_write_table(results["draws"], outdir / "posterior_draws"))
    written["event_summaries"] = str(_write_table(results["summary"], outdir / "event_summaries"))
    written["event_failures"] = str(_write_table(results["failures"], outdir / "event_failures"))
    written["event_covariances"] = str(_write_table(results["cov"], outdir / "event_covariances"))
    written["event_correlations"] = str(_write_table(results["corr"], outdir / "event_correlations"))
    written["branch_probabilities"] = str(_write_table(results["branch"], outdir / "branch_probabilities"))
    written["regime_probabilities"] = str(_write_table(results["regime"], outdir / "regime_probabilities"))
    written["event_metadata"] = str(_write_table(event_metadata_df, outdir / "event_metadata"))
    written["hitter_constants"] = str(_write_table(hitter_constants_df, outdir / "hitter_constants"))
    written["hitter_order"] = str(_write_table(hitter_order, outdir / "hitter_order"))
    written["hitter_meta"] = str(_write_table(hitter_meta, outdir / "hitter_meta"))

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "sensor_sigmas": sensor_sigmas,
        "success_count": results["success_count"],
        "processed_count": results["processed_count"],
        "exhausted_dataset": results["exhausted_dataset"],
        "files": written,
        "notes": {
            "z_definition": ["z_x_in", "z_theta_deg", "z_phi_deg"],
            "u_definition": ["u_bat_speed_mph", "u_attack_angle_deg", "u_attack_direction_deg"],
            "full_inverse_latents": ["latent_psi_abs_deg", "latent_psi_rel_deg", "latent_e_x", "latent_e_y", "latent_omega_plus", "latent_omega_minus"],
            "player_level_posterior_for_x_cm_and_r_g": "Not available in current code path. Fixed values are exported instead.",
            "posterior_entropy": "Gaussian plug-in approximation based on event-level covariance over selected continuous latent variables.",
            "uncertainty_volume": "sqrt(det(cov)) plug-in metric.",
            "branch_probabilities": "Posterior frequencies of sampled admissible intervals, plus average map branch probabilities when available.",
            "sensor_stability": "Approximated using per-draw map-weighted moments under the denoised sensor layer visited by the chain.",
        },
    }
    written["run_metadata"] = str(_write_json(metadata, outdir / "run_metadata.json"))

    print("\nRun complete.")
    print(json.dumps({
        "success_count": results["success_count"],
        "processed_count": results["processed_count"],
        "exhausted_dataset": results["exhausted_dataset"],
        "output_dir": str(outdir),
    }, indent=2))


import numpy as np
import pandas as pd

# =========================================================
# USER SETTINGS / TOGGLES
# =========================================================

USE_FAIR_ONLY = True          # True = require forward field geometry (hc_y_field > 0)
MIN_LAUNCH_SPEED_MPH = 40.0   # sample-definition filter, not strictly sensor sanity
MIN_BAT_SPEED_MPH = 40.0      # sample-definition filter, not strictly sensor sanity

WOOD_IZ_OZ_IN2 = 9.5
BALL_RADIUS_IN = 1.42
BALL_MASS_OZ = 5.04
ALPHA = 0.4

MPH_TO_FPS = 5280.0 / 3600.0
FPS_TO_MPH = 3600.0 / 5280.0
RPM_TO_RAD_S = 2.0 * np.pi / 60.0

# Spin-proxy settings
ASR_DEFAULT = 1.0             # active spin ratio fallback
SPIN_TORQUE_K = 0.02          # Nathan-style spin-down parameter
SPIN_AXIS_OFFSET_DEG = 0.0    # use if you later detect a constant axis offset
SPIN_SIGN_CORRECTION = 1.0    # set to -1.0 if your global sign is flipped

HC_HOME_X = 125.42
HC_HOME_Y = 198.27

# =========================================================
# BAT RADIUS PROFILE FOR STANDARD WOOD BAT
# =========================================================
# x is distance from knob in inches
# returns local bat radius R(x) in inches

WOOD_PROFILE_X_CM = np.array([0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80], dtype=float)
WOOD_PROFILE_R_MM = np.array([25, 17, 15, 13.5, 13.0, 13.2, 14.0, 15.5, 20.0, 28.0, 32.0, 33.0], dtype=float)



# =========================================================
# HITTER CLASS
# =========================================================

class Hitter:
    def __init__(
        self,
        name,
        bat_length,
        bat_weight,
        x_cm_fixed,
        r_g_fixed,
        r_ball_in=BALL_RADIUS_IN,
        alpha=ALPHA,
        Iz_oz_in2=WOOD_IZ_OZ_IN2,
        ball_mass_oz=BALL_MASS_OZ
    ):
        self.name = name
        self.bat_length = float(bat_length)      # L in inches
        self.bat_weight_oz = float(bat_weight)   # M in oz
        self.x_cm = float(x_cm_fixed)            # in
        self.r_g = float(r_g_fixed)              # in

        # Exogenous constants in baseball units
        self.r_ball_in = float(r_ball_in)
        self.alpha = float(alpha)
        self.Iz_oz_in2 = float(Iz_oz_in2)
        self.m_ball_oz = float(ball_mass_oz)

        # Derived inertia in baseball units
        self.I0_oz_in2 = self.bat_weight_oz * (self.r_g ** 2)

        # SI versions if needed elsewhere
        self.M_kg = self.bat_weight_oz * 0.0283495
        self.m_ball_kg = self.m_ball_oz * 0.0283495


        # sampled root coordinate placeholder
        self.x = None

    def sample_priors(self):
        """
        Replace with your actual x prior logic.
        For now, sample uniformly over the last 11 inches of the bat.
        """
        self.x = np.random.uniform(self.bat_length - 11.0, self.bat_length)

    def b(self, x_in=None):
        """
        Return b(x) = x - x_cm in inches.
        If x_in is None, use the hitter's current sampled x.
        """
        if x_in is None:
            if self.x is None:
                raise ValueError(f"{self.name}: x has not been sampled yet.")
            x_in = self.x
        return float(x_in - self.x_cm)

    def b_at_x(self, x_in):
        return self.b(x_in)

    @staticmethod
    def R(x, L=33.0, clip=True, geometric_similarity=False):
        """
        Local bat radius profile for a bat of length L, using the 33-inch
        reference profile with axial rescaling.
    
        Parameters
        ----------
        x : float or array-like
            Distance from knob in inches on the actual bat.
        L : float
            Bat length in inches.
        clip : bool
            If True, clip to the supported interval of the reference profile.
        geometric_similarity : bool
            If True, also scale the radius magnitude by L / 33.
            If False, only rescale position along the bat.
    
        Returns
        -------
        float or np.ndarray
            R(x; L) in inches.
        """
        x = np.asarray(x, dtype=float)
        x_ref = REF_LEN_IN * x / float(L)
    
        if clip:
            x_ref = np.clip(x_ref, 0.0, 33.0)
    
        y = np.full_like(x_ref, np.nan, dtype=float)
    
        # Interval 1: x_ref in [0, 2.0625]
        m = (x_ref >= 0.0) & (x_ref <= 2.0625)
        dx = x_ref[m] - 0.0
        y[m] = (
            -0.0387062012796 * dx**3
            + 0.241263088148 * dx**2
            - 0.526975964567 * dx
            + 0.977654677481
        )
    
        # Interval 2: x_ref in [2.0625, 4.125]
        m = (x_ref > 2.0625) & (x_ref <= 4.125)
        dx = x_ref[m] - 2.0625
        y[m] = (
            0.000243753253669 * dx**3
            + 0.00176846773047 * dx**2
            - 0.0257233805676 * dx
            + 0.577481421583
        )
    
        # Interval 3: x_ref in [4.125, 12.375]
        m = (x_ref > 4.125) & (x_ref <= 12.375)
        dx = x_ref[m] - 4.125
        y[m] = (
            -0.000140760341179 * dx**3
            + 0.00327669098755 * dx**2
            - 0.0153177407117 * dx
            + 0.53408845854
        )
    
        # Interval 4: x_ref in [12.375, 20.625]
        m = (x_ref > 12.375) & (x_ref <= 20.625)
        dx = x_ref[m] - 12.375
        y[m] = (
            0.000297591060378 * dx**3
            - 0.000207127456646 * dx**2
            + 0.0100061584182 * dx
            + 0.551697747056
        )
    
        # Interval 5: x_ref in [20.625, 24.75]
        m = (x_ref > 20.625) & (x_ref <= 24.75)
        dx = x_ref[m] - 20.625
        y[m] = (
            -0.00120367988811 * dx**3
            + 0.00715825128771 * dx**2
            + 0.0673529300245 * dx
            + 0.787252971751
        )
    
        # Interval 6: x_ref in [24.75, 28.875]
        m = (x_ref > 24.75) & (x_ref <= 28.875)
        dx = x_ref[m] - 24.75
        y[m] = (
            0.000300969030916 * dx**3
            - 0.00773728732763 * dx**2
            + 0.0649644063598 * dx
            + 1.10240029459
        )
    
        # Interval 7: x_ref in [28.875, 33.0]
        m = (x_ref > 28.875) & (x_ref <= 33.0)
        dx = x_ref[m] - 28.875
        y[m] = (
            0.000564165533097 * dx**3
            - 0.00401279557005 * dx**2
            + 0.0164953144069 * dx
            + 1.25984854282
        )
    
        if geometric_similarity:
            y = y * (float(L) / REF_LEN_IN)

        return y.item() if y.ndim == 0 else y

    def R_at_x(self, x_in, clip=True, geometric_similarity=False):
        return Hitter.R(x_in, L=self.bat_length, clip=clip, geometric_similarity=geometric_similarity)

    def R_player(x_in, hitter, clip=True):
        return hitter.R_at_x(x_in, clip=clip)


    def recoil_factors(self, x_in=None):
        """
        Return ry(x), rx(x) using baseball-unit formulas.
        """
        if x_in is None:
            if self.x is None:
                raise ValueError(f"{self.name}: x has not been sampled yet.")
            x_in = self.x

        b_val = self.b(x_in)
        R_val = self.R_at_x(x_in)

        ry = self.m_ball_oz * (
            1.0 / self.bat_weight_oz + (b_val ** 2) / self.I0_oz_in2
        )

        rx = self.m_ball_oz * self.alpha / (1.0 + self.alpha) * (
            1.0 / self.bat_weight_oz
            + (b_val ** 2) / self.I0_oz_in2
            + (R_val ** 2) / self.Iz_oz_in2
        )

        return ry, rx

# =========================================================
# FIXED BAT GEOMETRY LOOKUP
# =========================================================

fixed_param_lookup = {
    "aaron judge":       {"x_cm_fixed": 22.3109, "r_g_fixed":  9.6651},
    "mookie betts":      {"x_cm_fixed": 20.7747, "r_g_fixed":  9.1964},
    "pete alonso":       {"x_cm_fixed": 21.5354, "r_g_fixed":  9.5466},
    "george springer":   {"x_cm_fixed": 21.2330, "r_g_fixed":  9.2987},
    "jose altuve":       {"x_cm_fixed": 20.7169, "r_g_fixed":  9.1572},
    "ozzie albies":      {"x_cm_fixed": 21.5227, "r_g_fixed":  9.5879},
    "kike hernandez":    {"x_cm_fixed": 21.7372, "r_g_fixed":  9.8982},
    "bobby witt jr.":    {"x_cm_fixed": 21.9286, "r_g_fixed":  9.6180},
    "julio rodriguez":   {"x_cm_fixed": 21.4433, "r_g_fixed":  9.8156},
    "luis robert":       {"x_cm_fixed": 20.9189, "r_g_fixed":  9.3208},
    "manny machado":     {"x_cm_fixed": 21.8423, "r_g_fixed":  9.8509},
    "ronald acuna jr.":  {"x_cm_fixed": 21.1581, "r_g_fixed":  9.5229},
    "alex bregman":      {"x_cm_fixed": 21.2721, "r_g_fixed":  9.3348},
    "mike trout":        {"x_cm_fixed": 21.2227, "r_g_fixed":  9.2973},
    "nolan arenado":     {"x_cm_fixed": 21.2614, "r_g_fixed":  9.8307},
    "salvador perez":    {"x_cm_fixed": 20.7344, "r_g_fixed":  9.1710},
    "giancarlo stanton": {"x_cm_fixed": 21.7430, "r_g_fixed":  9.7644},
    "kris bryant":       {"x_cm_fixed": 21.0904, "r_g_fixed":  9.3764},
    "evan longoria":     {"x_cm_fixed": 21.1792, "r_g_fixed":  9.2690},
    "franmil reyes":     {"x_cm_fixed": 22.3134, "r_g_fixed":  9.9027},
    "luke voit":         {"x_cm_fixed": 22.2528, "r_g_fixed": 10.0225},
}

# =========================================================
# HITTER BAT SPECS
# =========================================================

hitter_specs = [
    ("aaron judge", 35.0, 33.0),
    ("mookie betts", 33.5, 31.5),
    ("pete alonso", 34.0, 32.0),
    ("george springer", 33.5, 30.5),
    ("jose altuve", 33.0, 31.0),
    ("ozzie albies", 34.5, 32.0),
    ("kike hernandez", 33.5, 31.0),
    ("bobby witt jr.", 33.5, 31.0),
    ("julio rodriguez", 34.0, 31.0),
    ("luis robert", 34.0, 31.0),
    ("manny machado", 34.0, 32.0),
    ("ronald acuna jr.", 33.5, 30.5),
    ("alex bregman", 33.5, 31.0),
    ("mike trout", 33.5, 31.5),
    ("nolan arenado", 34.0, 31.5),
    ("salvador perez", 33.75, 31.0),
    ("giancarlo stanton", 34.0, 32.0),
    ("kris bryant", 34.0, 31.5),
    ("evan longoria", 33.0, 31.0),
    ("franmil reyes", 34.0, 31.0),
    ("luke voit", 34.0, 31.0),

    # Left-handed hitters, add back if desired:
    # ("spencer horwitz", 33.5, 31.0),
    # ("ivan herrera", 33.5, 31.0),
    # ("yohel pozo", 34.0, 31.0),
    # ("bryce harper", 34.0, 32.0),
    # ("kyle schwarber", 34.0, 31.0),
    # ("juan soto", 34.0, 31.0),
    # ("rafael devers", 33.5, 31.5),
    # ("shohei ohtani", 34.5, 32.0),
    # ("adley rutschman", 33.5, 31.0),
    # ("jp crawford", 33.5, 31.0),
    # ("oneil cruz", 34.0, 31.0),
    # ("jake cronenworth", 33.5, 31.0),
    # ("kyle tucker", 33.5, 30.5),
    # ("yordan alvarez", 34.0, 31.0),
    # ("corey seager", 34.0, 32.0),
    # ("anthony rizzo", 34.5, 31.8),
]

# =========================================================
# SANITY CHECK: EVERY HITTER SPEC MUST HAVE FIXED PARAMS
# =========================================================

missing_lookup = [name for name, _, _ in hitter_specs if name not in fixed_param_lookup]
if missing_lookup:
    raise ValueError(f"Missing fixed_param_lookup entries for: {missing_lookup}")

# =========================================================
# CREATE HITTER OBJECTS
# =========================================================



def build_hitters_and_bip_model(batter_data_ext, verbose=True):
    _print = print if verbose else (lambda *args, **kwargs: None)
    _display = display if verbose else (lambda *args, **kwargs: None)
    hitters = []
    for name, bat_length, bat_weight in hitter_specs:
        params = fixed_param_lookup[name]
        hitters.append(
            Hitter(
                name=name,
                bat_length=bat_length,
                bat_weight=bat_weight,
                x_cm_fixed=params["x_cm_fixed"],
                r_g_fixed=params["r_g_fixed"],
                r_ball_in=BALL_RADIUS_IN,
                alpha=ALPHA,
                Iz_oz_in2=WOOD_IZ_OZ_IN2,
                ball_mass_oz=BALL_MASS_OZ,
            )
        )

    for h in hitters:
        h.sample_priors()
    # =========================================================
    # BUILD HITTER METADATA
    # =========================================================

    hitter_meta = pd.DataFrame([
        {
            "batter_name": h.name,
            "L_in": h.bat_length,
            "W_oz": h.bat_weight_oz,
            "M_kg": h.M_kg,
            "m_ball_kg": h.m_ball_kg,
            "m_ball_oz": h.m_ball_oz,
            "r_ball_in": h.r_ball_in,
            "alpha": h.alpha,
            "x_prior_sample_in": h.x,
            "b_prior_sample_in": h.b(),
            "x_cm_fixed_in": h.x_cm,
            "r_g_fixed_in": h.r_g,
            "I0_oz_in2": h.I0_oz_in2,
            "Iz_oz_in2": h.Iz_oz_in2,
        }
        for h in hitters
    ])

    # =========================================================
    # BUILD CLEAN BIP DATASET FROM batter_data_ext
    # =========================================================

    raw_cols = [
        # identifiers / context
        "game_date", "batter", "batter_name", "pitcher", "pitch_number",
        "pitch_type", "stand", "p_throws", "balls", "strikes",
        "type", "events", "description", "bb_type", 

        # raw pitch / trajectory inputs
        "vx0", "vy0", "vz0",
        "ax", "ay", "az",
        "release_pos_x", "release_pos_y", "release_pos_z", "plate_x", "plate_z",

        # batted-ball observations
        "launch_speed", "launch_angle", "hc_x", "hc_y",

        # bat observations
        "bat_speed", "attack_angle", "attack_direction",

        # pitch spin inputs for omega^- proxy
        "release_spin_rate", "spin_axis",
    ]

    launch_speed_num = pd.to_numeric(batter_data_ext["launch_speed"], errors="coerce")
    bat_speed_num = pd.to_numeric(batter_data_ext["bat_speed"], errors="coerce")

    bip_df = (
        batter_data_ext.loc[
            (batter_data_ext["type"] == "X") &
            (launch_speed_num >= MIN_LAUNCH_SPEED_MPH) &
            (bat_speed_num >= MIN_BAT_SPEED_MPH),
            raw_cols
        ]
        .copy()
    )

    # Convert numeric columns once
    numeric_cols = [
        "vx0", "vy0", "vz0",
        "ax", "ay", "az",
        "release_pos_x", "release_pos_y", "release_pos_z",
        "launch_speed", "launch_angle", "hc_x", "hc_y",
        "bat_speed", "attack_angle", "attack_direction",
        "release_spin_rate", "spin_axis", "plate_x", "plate_z"
    ]

    for c in numeric_cols:
        bip_df[c] = pd.to_numeric(bip_df[c], errors="coerce")

    # Drop rows missing critical reconstruction fields
    # Keep spin fields in here because omega^- is part of the manifold input.
    bip_df = bip_df.dropna(subset=[
        "vx0", "vy0", "vz0",
        "ax", "ay", "az",
        "release_pos_y",
        "launch_speed", "launch_angle", "hc_x", "hc_y",
        "bat_speed", "attack_angle", "attack_direction",
        "release_spin_rate", "spin_axis",
    ]).copy()

    # Unit conversions
    bip_df["launch_speed_fps"] = bip_df["launch_speed"] * MPH_TO_FPS
    bip_df["bat_speed_fps"] = bip_df["bat_speed"] * MPH_TO_FPS

    bip_df["launch_angle_rad"] = np.deg2rad(bip_df["launch_angle"])
    bip_df["attack_angle_rad"] = np.deg2rad(bip_df["attack_angle"])
    bip_df["attack_direction_rad"] = np.deg2rad(bip_df["attack_direction"])

    # =========================================================
    # FILTER TO TARGET HITTERS AND MERGE METADATA
    # =========================================================

    bip_sub = (
        bip_df[bip_df["batter_name"].isin(hitter_meta["batter_name"])]
        .merge(hitter_meta, on="batter_name", how="left")
        .copy()
    )

    # =========================================================
    # REQUIRED INPUTS FLAG
    # =========================================================

    required_cols = [
        "bat_speed_fps",
        "attack_angle_rad",
        "attack_direction_rad",
        "vx0", "vy0", "vz0",
        "ax", "ay", "az",
        "release_pos_y",
        "launch_speed_fps",
        "launch_angle_rad",
        "hc_x", "hc_y",
        "release_spin_rate",
        "spin_axis",
        "plate_x",
        "plate_z"
    ]

    bip_sub["flag_missing_required"] = bip_sub[required_cols].isna().any(axis=1)

    # =========================================================
    # BAT VELOCITY VECTOR  v_b^-
    # =========================================================

    aa = bip_sub["attack_angle_rad"]
    ad = bip_sub["attack_direction_rad"]
    vb = bip_sub["bat_speed_fps"]

    bip_sub["vbat_x"] = vb * np.cos(aa) * np.sin(ad)
    bip_sub["vbat_y"] = vb * np.cos(aa) * np.cos(ad)
    bip_sub["vbat_z"] = vb * np.sin(aa)

    bip_sub["vbat_mag"] = np.sqrt(
        bip_sub["vbat_x"]**2 +
        bip_sub["vbat_y"]**2 +
        bip_sub["vbat_z"]**2
    )

    # =========================================================
    # SPRAY ANGLE  phi  FROM hc_x, hc_y
    # =========================================================

    bip_sub["hc_x_field"] = bip_sub["hc_x"] - HC_HOME_X
    bip_sub["hc_y_field"] = HC_HOME_Y - bip_sub["hc_y"]

    # Spray angle from +y toward +x
    bip_sub["spray_phi_rad"] = np.arctan2(
        bip_sub["hc_x_field"],
        bip_sub["hc_y_field"]
    )
    bip_sub["spray_angle_deg"] = np.degrees(bip_sub["spray_phi_rad"])

    # =========================================================
    # OBSERVED OUTGOING DIRECTION  vhat_obs
    # =========================================================
    # rhat(phi) = (sin phi, cos phi, 0)
    # vhat_obs = cos(theta) rhat(phi) + sin(theta) e_z

    theta = bip_sub["launch_angle_rad"]
    phi = bip_sub["spray_phi_rad"]
    s = bip_sub["launch_speed_fps"]

    bip_sub["rhat_x"] = np.sin(phi)
    bip_sub["rhat_y"] = np.cos(phi)
    bip_sub["rhat_z"] = 0.0

    bip_sub["vhat_obs_x"] = np.cos(theta) * bip_sub["rhat_x"]
    bip_sub["vhat_obs_y"] = np.cos(theta) * bip_sub["rhat_y"]
    bip_sub["vhat_obs_z"] = np.sin(theta)

    bip_sub["vhat_obs_mag"] = np.sqrt(
        bip_sub["vhat_obs_x"]**2 +
        bip_sub["vhat_obs_y"]**2 +
        bip_sub["vhat_obs_z"]**2
    )

    # =========================================================
    # OBSERVED OUTGOING BALL VELOCITY VECTOR  v_B^+
    # =========================================================

    bip_sub["vobs_x"] = s * bip_sub["vhat_obs_x"]
    bip_sub["vobs_y"] = s * bip_sub["vhat_obs_y"]
    bip_sub["vobs_z"] = s * bip_sub["vhat_obs_z"]

    bip_sub["vobs_mag"] = np.sqrt(
        bip_sub["vobs_x"]**2 +
        bip_sub["vobs_y"]**2 +
        bip_sub["vobs_z"]**2
    )

    bip_sub["horizontal_exit_speed"] = np.sqrt(
        bip_sub["vobs_x"]**2 + bip_sub["vobs_y"]**2
    )

    # =========================================================
    # INCOMING PITCH VELOCITY AT CONTACT  v_B^-
    # =========================================================

    A = 0.5 * bip_sub["ay"]
    B = bip_sub["vy0"]
    C = bip_sub["release_pos_y"]

    disc = B**2 - 4 * A * C
    bip_sub["t_contact"] = np.nan

    eps = 1e-8

    use_quad = (disc >= 0) & (A.abs() > eps) & B.notna() & C.notna()
    sqrt_disc = np.sqrt(disc[use_quad])

    t1 = (-B[use_quad] + sqrt_disc) / (2 * A[use_quad])
    t2 = (-B[use_quad] - sqrt_disc) / (2 * A[use_quad])

    t_candidates = np.vstack([t1.to_numpy(), t2.to_numpy()]).T
    t_pos = np.where(t_candidates > 0, t_candidates, np.inf)
    t_choice = np.min(t_pos, axis=1)
    t_choice[np.isinf(t_choice)] = np.nan

    bip_sub.loc[use_quad, "t_contact"] = t_choice

    use_lin = bip_sub["t_contact"].isna() & (B.abs() > eps) & B.notna() & C.notna()
    bip_sub.loc[use_lin, "t_contact"] = -C[use_lin] / B[use_lin]

    bip_sub.loc[bip_sub["t_contact"] <= 0, "t_contact"] = np.nan

    t = bip_sub["t_contact"]

    bip_sub["vin_x"] = bip_sub["vx0"] + bip_sub["ax"] * t
    bip_sub["vin_y"] = bip_sub["vy0"] + bip_sub["ay"] * t
    bip_sub["vin_z"] = bip_sub["vz0"] + bip_sub["az"] * t

    bip_sub["vin_mag"] = np.sqrt(
        bip_sub["vin_x"]**2 +
        bip_sub["vin_y"]**2 +
        bip_sub["vin_z"]**2
    )

    # Keep alias if you prefer the older name
    bip_sub["vin_speed"] = bip_sub["vin_mag"]

    # =========================================================
    # OPTIONAL PRE-COLLISION RELATIVE VELOCITY
    # =========================================================
    # Using bat minus ball, consistent with Delta = v_b^- - v_B^- later.

    bip_sub["vrel_in_x"] = bip_sub["vbat_x"] - bip_sub["vin_x"]
    bip_sub["vrel_in_y"] = bip_sub["vbat_y"] - bip_sub["vin_y"]
    bip_sub["vrel_in_z"] = bip_sub["vbat_z"] - bip_sub["vin_z"]

    bip_sub["vrel_in_mag"] = np.sqrt(
        bip_sub["vrel_in_x"]**2 +
        bip_sub["vrel_in_y"]**2 +
        bip_sub["vrel_in_z"]**2
    )

    # =========================================================
    # RELEVANT PRE-COLLISION SPIN COMPONENT  omega^-
    # =========================================================

    def add_relevant_spin_component(
        df,
        k_torque=SPIN_TORQUE_K,
        asr_default=ASR_DEFAULT,
        asr_col=None,
        n_steps=12,
        spin_axis_offset_deg=SPIN_AXIS_OFFSET_DEG,
        sign_correction=SPIN_SIGN_CORRECTION,
    ):
        """
        Adds the relevant pre-collision spin component omega_minus for the coplanar manifold.

        Assumption:
          - spin_axis is used as a directional proxy for the active-spin axis
          - active spin vector is approximated in the x-z plane
          - spin decays during flight according to a Nathan-style exponential model

        Outputs:
          - chat_x, chat_y, chat_z
          - omega_release_rpm
          - omega_active_release_rpm
          - omega_contact_rpm
          - omega_vec_contact_{x,y,z}_rpm
          - omega_minus_rpm
          - omega_minus_rad_s
        """
        out = df.copy()

        # c-hat(phi) = r-hat(phi) x e_z = (cos phi, -sin phi, 0)
        phi = out["spray_phi_rad"]
        out["chat_x"] = np.cos(phi)
        out["chat_y"] = -np.sin(phi)
        out["chat_z"] = 0.0

        # Release spin magnitude
        out["omega_release_rpm"] = out["release_spin_rate"].astype(float)

        if asr_col is not None and asr_col in out.columns:
            asr = out[asr_col].astype(float).clip(lower=0.0, upper=1.0)
        else:
            asr = pd.Series(asr_default, index=out.index, dtype=float)

        out["active_spin_ratio_used"] = asr
        out["omega_active_release_rpm"] = out["omega_release_rpm"] * asr

        # Spin-axis mapping proxy
        axis_deg = out["spin_axis"].astype(float) + spin_axis_offset_deg
        axis_rad = np.deg2rad(axis_deg)

        out["omega_hat_x"] = np.cos(axis_rad)
        out["omega_hat_y"] = 0.0
        out["omega_hat_z"] = np.sin(axis_rad)

        # Spin-down during flight
        t_contact = out["t_contact"].astype(float).clip(lower=0.0)
        grid = np.linspace(0.0, 1.0, n_steps)
        T = t_contact.to_numpy()[:, None] * grid[None, :]

        vx = out["vx0"].to_numpy()[:, None] + out["ax"].to_numpy()[:, None] * T
        vy = out["vy0"].to_numpy()[:, None] + out["ay"].to_numpy()[:, None] * T
        vz = out["vz0"].to_numpy()[:, None] + out["az"].to_numpy()[:, None] * T

        v_fps = np.sqrt(vx**2 + vy**2 + vz**2)
        v_mph = v_fps * FPS_TO_MPH

        int_v_dt = np.trapezoid(v_mph, T, axis=1)
        decay_factor = np.exp(-0.020 * k_torque * int_v_dt)

        out["spin_decay_factor"] = decay_factor
        out["omega_contact_rpm"] = out["omega_active_release_rpm"] * decay_factor

        # Contact-time active spin vector
        out["omega_vec_contact_x_rpm"] = out["omega_contact_rpm"] * out["omega_hat_x"]
        out["omega_vec_contact_y_rpm"] = out["omega_contact_rpm"] * out["omega_hat_y"]
        out["omega_vec_contact_z_rpm"] = out["omega_contact_rpm"] * out["omega_hat_z"]

        # Relevant coplanar spin component
        omega_dot_chat_rpm = (
            out["omega_vec_contact_x_rpm"] * out["chat_x"] +
            out["omega_vec_contact_y_rpm"] * out["chat_y"] +
            out["omega_vec_contact_z_rpm"] * out["chat_z"]
        )

        out["omega_minus_rpm"] = sign_correction * (-omega_dot_chat_rpm)
        out["omega_minus_rad_s"] = out["omega_minus_rpm"] * RPM_TO_RAD_S

        return out

    bip_sub = add_relevant_spin_component(bip_sub)

    # =========================================================
    # SENSOR / INPUT SANITY FLAGS
    # =========================================================

    bip_sub["flag_bad_time"] = bip_sub["t_contact"].isna() | (bip_sub["t_contact"] > 0.75)
    bip_sub["flag_bad_vin_y"] = bip_sub["vin_y"] >= 0

    bip_sub["flag_bad_bat_speed"] = (
        bip_sub["bat_speed_fps"].isna() |
        (bip_sub["bat_speed_fps"] < 10.0) |
        (bip_sub["bat_speed_fps"] > 160.0)
    )

    bip_sub["flag_bad_exit_speed"] = (
        bip_sub["launch_speed_fps"].isna() |
        (bip_sub["launch_speed_fps"] <= 0.0) |
        (bip_sub["launch_speed_fps"] > 190.0)
    )

    bip_sub["flag_bad_launch_angle"] = (
        bip_sub["launch_angle"].isna() |
        (bip_sub["launch_angle"] < -90.0) |
        (bip_sub["launch_angle"] > 90.0)
    )

    bip_sub["flag_bad_attack_angle"] = (
        bip_sub["attack_angle"].isna() |
        (bip_sub["attack_angle"] < -50) |
        (bip_sub["attack_angle"] > 50)
    )

    bip_sub["flag_bad_attack_direction"] = (
        bip_sub["attack_direction"].isna() |
        (bip_sub["attack_direction"] < -90) |
        (bip_sub["attack_direction"] > 90)
    )

    bip_sub["flag_bad_spin_rate"] = (
        bip_sub["release_spin_rate"].isna() |
        (bip_sub["release_spin_rate"] <= 0.0) |
        (bip_sub["release_spin_rate"] > 5000.0)
    )

    bip_sub["flag_bad_spin_axis"] = (
        bip_sub["spin_axis"].isna() |
        (bip_sub["spin_axis"] < 0.0) |
        (bip_sub["spin_axis"] > 360.0)
    )

    bip_sub["flag_bad_omega_minus"] = bip_sub["omega_minus_rad_s"].isna()

    # =========================================================
    # FIELD GEOMETRY FLAGS
    # =========================================================

    # Forward field means out in front of home plate
    bip_sub["flag_forward_field"] = bip_sub["hc_y_field"] > 0
    bip_sub["flag_backward_field"] = ~bip_sub["flag_forward_field"]

    # Inside fair wedge means |x| <= y with y > 0
    bip_sub["flag_inside_fair_wedge"] = (
        bip_sub["flag_forward_field"] &
        (np.abs(bip_sub["hc_x_field"]) <= bip_sub["hc_y_field"])
    )

    bip_sub["flag_outside_fair_wedge"] = ~bip_sub["flag_inside_fair_wedge"]

    # =========================================================
    # SANITY CHECKS
    # =========================================================

    _print("\n================ SANITY CHECKS ================\n")
    _print("Rows in bip_sub:", len(bip_sub))
    _print("Hitters in bip_sub:", bip_sub["batter_name"].nunique())

    _print("\nUnit-vector check for observed outgoing direction:")
    _display((bip_sub["vhat_obs_mag"] - 1.0).abs().describe())

    _print("\nBat speed reconstruction error:")
    _display((bip_sub["vbat_mag"] - bip_sub["bat_speed_fps"]).abs().describe())

    _print("\nExit speed reconstruction error:")
    _display((bip_sub["vobs_mag"] - bip_sub["launch_speed_fps"]).abs().describe())

    _print("\nHorizontal exit speed reconstruction error:")
    _display(
        (
            bip_sub["horizontal_exit_speed"]
            - bip_sub["launch_speed_fps"] * np.cos(bip_sub["launch_angle_rad"])
        ).abs().describe()
    )

    _print("\nIncoming pitch direction:")
    _print("Fraction vy0 < 0:", (bip_sub["vy0"] < 0).mean())
    _print("Fraction vin_y < 0 at contact:", (bip_sub["vin_y"] < 0).mean())

    _print("\nSpray angle (deg):")
    _display(bip_sub["spray_angle_deg"].describe())

    _print("\nSpin decay factor:")
    _display(bip_sub["spin_decay_factor"].describe())

    _print("\nomega_minus (rpm):")
    _display(bip_sub["omega_minus_rpm"].describe())

    _print("\nFlag rates:")
    flag_cols = [
        "flag_missing_required",
        "flag_bad_time",
        "flag_bad_vin_y",
        "flag_bad_bat_speed",
        "flag_bad_exit_speed",
        "flag_bad_launch_angle",
        "flag_bad_attack_angle",
        "flag_bad_attack_direction",
        "flag_bad_spin_rate",
        "flag_bad_spin_axis",
        "flag_bad_omega_minus",
        "flag_backward_field",
        "flag_outside_fair_wedge",
    ]
    _display(bip_sub[flag_cols].mean().sort_values(ascending=False))

    # =========================================================
    # BUILD MANIFOLD DATASET
    # =========================================================

    base_mask = (
        (~bip_sub["flag_missing_required"]) &
        (~bip_sub["flag_bad_time"]) &
        (~bip_sub["flag_bad_vin_y"]) &
        (~bip_sub["flag_bad_bat_speed"]) &
        (~bip_sub["flag_bad_exit_speed"]) &
        (~bip_sub["flag_bad_launch_angle"]) &
        (~bip_sub["flag_bad_attack_angle"]) &
        (~bip_sub["flag_bad_attack_direction"]) &
        (~bip_sub["flag_bad_spin_rate"]) &
        (~bip_sub["flag_bad_spin_axis"]) &
        (~bip_sub["flag_bad_omega_minus"])
    )

    if USE_FAIR_ONLY:
        base_mask = base_mask & (~bip_sub["flag_outside_fair_wedge"])

    # =========================================================
    # CONTINUOUS COUNT FEATURE
    # =========================================================

    z_count_map = {
        "0-0": 0.3144746543,
        "0-1": 0.2704261285,
        "0-2": 0.2004055518,
        "1-0": 0.3612777549,
        "1-1": 0.3050139272,
        "1-2": 0.2260470281,
        "2-0": 0.4309212696,
        "2-1": 0.3613989554,
        "2-2": 0.2734161604,
        "3-0": 0.5416601816,
        "3-1": 0.4803511791,
        "3-2": 0.3824951644,
    }

    # build string count like "2-1"
    bip_sub["count_str"] = (
        bip_sub["balls"].astype("Int64").astype(str)
        + "-"
        + bip_sub["strikes"].astype("Int64").astype(str)
    )

    # map to continuous count value
    bip_sub["z_count"] = bip_sub["count_str"].map(z_count_map)

    # optional sanity check
    missing_counts = sorted(bip_sub.loc[bip_sub["z_count"].isna(), "count_str"].dropna().unique())
    if missing_counts:
        _print("Unmapped count values found:", missing_counts)

    bip_model = bip_sub.loc[base_mask].copy()

    _print("\nSpray angle (deg), fair wedge only:")
    fair_angles = bip_sub.loc[~bip_sub["flag_outside_fair_wedge"], "spray_angle_deg"]
    _display(fair_angles.describe())

    _print("Min fair spray angle:", fair_angles.min())
    _print("Max fair spray angle:", fair_angles.max())

    # =========================================================
    # ADD HITTER INDEX FOR HIERARCHICAL MODEL
    # =========================================================

    hitter_order = (
        bip_model[["batter_name"]]
        .drop_duplicates()
        .sort_values("batter_name")
        .reset_index(drop=True)
    )

    hitter_order["h_idx"] = np.arange(len(hitter_order), dtype=int)

    bip_model = bip_model.merge(hitter_order, on="batter_name", how="left")

    _print("\n================ MODEL DATASET SUMMARY ================\n")
    _print("Rows in bip_sub:", len(bip_sub))
    _print("Rows in bip_model:", len(bip_model))
    _print("Rows removed:", len(bip_sub) - len(bip_model))
    _print("Fraction kept:", len(bip_model) / len(bip_sub) if len(bip_sub) > 0 else np.nan)
    _print("Hitters kept in bip_model:", bip_model["batter_name"].nunique())
    _print("Any missing h_idx?", bip_model["h_idx"].isna().any())

    _print("\nHitter index mapping:")
    _display(hitter_order)
    return hitters, hitter_meta, bip_sub, bip_model, hitter_order


import numpy as np

# =========================================================
# BASIS VECTORS
# =========================================================

E_X = np.array([1.0, 0.0, 0.0])
E_Y = np.array([0.0, 1.0, 0.0])
E_Z = np.array([0.0, 0.0, 1.0])


# =========================================================
# SCALAR PHYSICS FUNCTIONS
# =========================================================

def b_of_x(x, x_cm):
    """
    b(x) = x - x_cm
    """
    return x - x_cm


def r_y_of_x(x, x_cm, m, M, I0):
    """
    r_y(x) = m * (1/M + b(x)^2 / I0)
    """
    b = b_of_x(x, x_cm)
    return m * (1.0 / M + (b ** 2) / I0)


def r_x_of_x(x, x_cm, m, alpha, M, I0, R, Iz):
    """
    r_x(x) = (m * alpha / (1 + alpha)) * (1/M + b(x)^2/I0 + R^2/Iz)

    Parameters
    ----------
    x : float or array
        Contact location along bat
    x_cm : float
        Bat center of mass location
    m : float
        Ball mass
    alpha : float
        Ball inertia coefficient (typically 0.4)
    M : float
        Bat mass
    I0 : float
        Bat MOI about knob / reference axis used in your equation
    R : float
        Local bat radius at contact
    Iz : float
        Bat axial MOI
    """
    b = b_of_x(x, x_cm)
    prefactor = m * alpha / (1.0 + alpha)
    return prefactor * (1.0 / M + (b ** 2) / I0 + (R ** 2) / Iz)


def D_of_psi(R, r, psi):
    """
    D = (R + r) * sin(psi)

    Parameters
    ----------
    R : float or array
        Local bat radius at contact
    r : float or array
        Ball radius
    psi : float or array
        Contact-plane angle
    """
    return (R + r) * np.sin(psi)


# =========================================================
# PLANAR UNIT VECTORS
# =========================================================

def rhat(phi):
    """
    rhat(phi) = sin(phi) e_x + cos(phi) e_y

    Returns a 3-vector or array of 3-vectors.
    """
    phi = np.asarray(phi)
    return np.stack(
        [np.sin(phi), np.cos(phi), np.zeros_like(phi)],
        axis=-1
    )


def chat(phi):
    """
    chat(phi) = rhat(phi) x e_z = cos(phi) e_x - sin(phi) e_y

    Returns a 3-vector or array of 3-vectors.
    """
    phi = np.asarray(phi)
    return np.stack(
        [np.cos(phi), -np.sin(phi), np.zeros_like(phi)],
        axis=-1
    )


def nhat(psi, phi):
    """
    nhat(psi, phi) = cos(psi) rhat(phi) + sin(psi) e_z
    """
    psi = np.asarray(psi)
    rh = rhat(phi)
    ez = np.broadcast_to(E_Z, rh.shape)
    return np.cos(psi)[..., None] * rh + np.sin(psi)[..., None] * ez


def that(psi, phi):
    """
    that(psi, phi) = -sin(psi) rhat(phi) + cos(psi) e_z
    """
    psi = np.asarray(psi)
    rh = rhat(phi)
    ez = np.broadcast_to(E_Z, rh.shape)
    return -np.sin(psi)[..., None] * rh + np.cos(psi)[..., None] * ez


# =========================================================
# OBSERVED OUTGOING DIRECTION / VELOCITY
# =========================================================

def vhat_obs(theta, phi):
    """
    vhat_obs = cos(theta) rhat(phi) + sin(theta) e_z
    """
    theta = np.asarray(theta)
    rh = rhat(phi)
    ez = np.broadcast_to(E_Z, rh.shape)
    return np.cos(theta)[..., None] * rh + np.sin(theta)[..., None] * ez


def vhat_obs_in_nt_basis(theta, psi, phi):
    """
    Returns the same observed unit vector expressed in the {nhat, that} basis:

    vhat_obs = cos(theta - psi) nhat(psi, phi) + sin(theta - psi) that(psi, phi)
    """
    theta = np.asarray(theta)
    psi = np.asarray(psi)

    nh = nhat(psi, phi)
    th = that(psi, phi)

    return (
        np.cos(theta - psi)[..., None] * nh +
        np.sin(theta - psi)[..., None] * th
    )


def V_n_plus(s, theta, psi):
    """
    V_n^+ = s * cos(theta - psi)
    """
    return s * np.cos(theta - psi)


def V_t_plus(s, theta, psi):
    """
    V_t^+ = s * sin(theta - psi)
    """
    return s * np.sin(theta - psi)


def v_obs_plus(s, theta, phi):
    """
    Full outgoing observed velocity vector:
    v_obs^+ = s * vhat_obs(theta, phi)
    """
    return np.asarray(s)[..., None] * vhat_obs(theta, phi)


# =========================================================
# GENERIC CHANGE OF BASIS: (u_r, u_z) -> (u_n, u_t)
# =========================================================

def u_n(u_r, u_z, psi):
    """
    u_n(psi, phi) = u_r(phi) cos(psi) + u_z sin(psi)

    Note:
    phi only enters through u_r(phi), so this function just takes u_r directly.
    """
    return u_r * np.cos(psi) + u_z * np.sin(psi)


def u_t(u_r, u_z, psi):
    """
    u_t(psi, phi) = -u_r(phi) sin(psi) + u_z cos(psi)
    """
    return -u_r * np.sin(psi) + u_z * np.cos(psi)


# =========================================================
# PRE-COLLISION BALL / BAT COMPONENTS
# =========================================================

def vB_n_minus(vB_r_minus, vB_z_minus, psi):
    """
    v_{B,n}^- = v_{B,r}^- cos(psi) + v_{B,z}^- sin(psi)
    """
    return u_n(vB_r_minus, vB_z_minus, psi)


def vB_t_minus(vB_r_minus, vB_z_minus, psi):
    """
    v_{B,t}^- = -v_{B,r}^- sin(psi) + v_{B,z}^- cos(psi)
    """
    return u_t(vB_r_minus, vB_z_minus, psi)


def vb_n_minus(vb_r_minus, vb_z_minus, psi):
    """
    v_{b,n}^- = v_{b,r}^- cos(psi) + v_{b,z}^- sin(psi)
    """
    return u_n(vb_r_minus, vb_z_minus, psi)


def vb_t_minus(vb_r_minus, vb_z_minus, psi):
    """
    v_{b,t}^- = -v_{b,r}^- sin(psi) + v_{b,z}^- cos(psi)
    """
    return u_t(vb_r_minus, vb_z_minus, psi)


def Delta_n(vb_r_minus, vb_z_minus, vB_r_minus, vB_z_minus, psi):
    """
    Delta_n = v_{b,n}^- - v_{B,n}^-
    """
    return (
        vb_n_minus(vb_r_minus, vb_z_minus, psi)
        - vB_n_minus(vB_r_minus, vB_z_minus, psi)
    )


def Delta_t(vb_r_minus, vb_z_minus, vB_r_minus, vB_z_minus, psi):
    """
    Delta_t = v_{b,t}^- - v_{B,t}^-
    """
    return (
        vb_t_minus(vb_r_minus, vb_z_minus, psi)
        - vB_t_minus(vB_r_minus, vB_z_minus, psi)
    )


# =========================================================
# OPTIONAL HELPERS: EXTRACT RADIAL COMPONENT FROM (x, y)
# =========================================================

def radial_component(u_x, u_y, phi):
    """
    u_r(phi) = u · rhat(phi) = u_x sin(phi) + u_y cos(phi)

    Use this if you start from Cartesian x/y components and need u_r(phi).
    """
    return u_x * np.sin(phi) + u_y * np.cos(phi)


def circumferential_component(u_x, u_y, phi):
    """
    u_c(phi) = u · chat(phi) = u_x cos(phi) - u_y sin(phi)

    Included because you'll probably need it later anyway.
    """
    return u_x * np.cos(phi) - u_y * np.sin(phi)

    import numpy as np


REF_LEN_IN = 33.0


def e_eff(x, L=33.0, clip=True):
    """
    Effective COR profile for a bat of length L, using the 33-inch
    reference profile with axial rescaling.

    Parameters
    ----------
    x : float or array-like
        Distance from knob in inches on the actual bat.
    L : float
        Bat length in inches.
    clip : bool
        If True, clip to the supported interval of the reference profile.

    Returns
    -------
    float or np.ndarray
        e_eff(x; L)
    """
    x = np.asarray(x, dtype=float)
    x_ref = REF_LEN_IN * x / float(L)

    if clip:
        x_ref = np.clip(x_ref, 15.7142857143, 33.0)

    y = np.full_like(x_ref, np.nan, dtype=float)

    # Interval 1: x_ref in [15.7142857143, 21.6071428571]
    m = (x_ref >= 15.7142857143) & (x_ref <= 21.6071428571)
    dx = x_ref[m] - 15.7142857143
    y[m] = (
        0.000717748009997 * dx**3
        - 0.00349607103449 * dx**2
        + 0.00956256809486 * dx
        + 0.14438264692
    )

    # Interval 2: x_ref in [21.6071428571, 26.7142857143]
    m = (x_ref > 21.6071428571) & (x_ref <= 26.7142857143)
    dx = x_ref[m] - 21.6071428571
    y[m] = (
        -0.001482259684 * dx**3
        + 0.00919268842795 * dx**2
        + 0.043131920592 * dx
        + 0.226205380091
    )

    # Interval 3: x_ref in [26.7142857143, 29.8571428571]
    m = (x_ref > 26.7142857143) & (x_ref <= 29.8571428571)
    dx = x_ref[m] - 26.7142857143
    y[m] = (
        0.000121796464513 * dx**3
        - 0.0135176474448 * dx**2
        + 0.0210437370416 * dx
        + 0.488807773322
    )

    # Interval 4: x_ref in [29.8571428571, 33.0]
    m = (x_ref > 29.8571428571) & (x_ref <= 33.0)
    dx = x_ref[m] - 29.8571428571
    y[m] = (
        0.00304528742295 * dx**3
        - 0.0123692807794 * dx**2
        - 0.0603151802346 * dx
        + 0.425204997059
    )

    return y.item() if y.ndim == 0 else y

# =========================================================
# FUNCTION BLOCK REPLACEMENT
# Exact 1D manifold in x:
#   x  -> psi_abs(x) -> {psi_rel, e_x, omega_plus}
#
# Key changes:
#   1) event cache now stores RAW sensor inputs only
#   2) m* is sampled in raw sensor space
#   3) all intermediate quantities are rebuilt for each m*
#   4) no branch machinery
#   5) D uses psi_rel directly and unambiguously
#   6) regime logic is hard admissibility:
#        - gross-slip => inadmissible
#        - stick-slip => e_x ~ N(0.4, 0.1)
#        - slip-stick-slip => e_x ~ Uniform(0, 0.6)
# =========================================================

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_DENOM_TOL = 1e-5
DEFAULT_ROOT_TOL = 1e-5
DEFAULT_MU = 0.5
DEFAULT_G2_BASEBALL = 4.75

DEFAULT_SIGMA_N = 0.1
DEFAULT_SIGMA_T = 0.1
DEFAULT_SIGMA_W = 0.1

# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def wrap_angle_pi(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_2pi(angle_rad):
    return angle_rad % (2.0 * np.pi)


def wrap_deg_360(angle_deg):
    return angle_deg % 360.0


def reflect_to_interval(x, lo, hi):
    if lo >= hi:
        raise ValueError("Invalid interval.")
    y = float(x)
    while (y < lo) or (y > hi):
        if y < lo:
            y = lo + (lo - y)
        if y > hi:
            y = hi - (y - hi)
    return y


def e_y_player(x_in, hitter, clip=True):
    return float(e_eff(x_in, L=hitter.bat_length, clip=clip))


def R_player(x_in, hitter, clip=True):
    return float(hitter.R_at_x(x_in, clip=clip))


# ---------------------------------------------------------
# Raw-sensor reconstruction
# ---------------------------------------------------------

def spray_phi_from_hc(hc_x, hc_y, home_x=HC_HOME_X, home_y=HC_HOME_Y):
    hc_x_field = float(hc_x) - float(home_x)
    hc_y_field = float(home_y) - float(hc_y)
    phi = np.arctan2(hc_x_field, hc_y_field)
    return float(phi), float(hc_x_field), float(hc_y_field)


def build_bat_velocity_from_sensor(bat_speed_mph, attack_angle_deg, attack_direction_deg):
    vb = float(bat_speed_mph) * MPH_TO_FPS
    aa = np.deg2rad(float(attack_angle_deg))
    ad = np.deg2rad(float(attack_direction_deg))

    vbat_x = vb * np.cos(aa) * np.sin(ad)
    vbat_y = vb * np.cos(aa) * np.cos(ad)
    vbat_z = vb * np.sin(aa)

    return {
        "bat_speed_fps": vb,
        "attack_angle_rad": aa,
        "attack_direction_rad": ad,
        "vbat_x": float(vbat_x),
        "vbat_y": float(vbat_y),
        "vbat_z": float(vbat_z),
    }


def solve_contact_time_from_sensor(release_pos_y, vy0, ay, max_t=0.75):
    y0 = float(release_pos_y)
    vy = float(vy0)
    acc = float(ay)

    # Solve y0 + vy t + 0.5 a t^2 = 0
    A = 0.5 * acc
    B = vy
    C = y0

    eps = 1e-10
    t_contact = np.nan

    if abs(A) > eps:
        disc = B * B - 4.0 * A * C
        if disc >= 0.0:
            sqrt_disc = np.sqrt(disc)
            roots = [(-B + sqrt_disc) / (2.0 * A), (-B - sqrt_disc) / (2.0 * A)]
            roots = [r for r in roots if r > 0.0]
            if roots:
                t_contact = min(roots)

    if not np.isfinite(t_contact) and abs(B) > eps:
        t_lin = -C / B
        if t_lin > 0.0:
            t_contact = t_lin

    if not np.isfinite(t_contact) or t_contact <= 0.0 or t_contact > max_t:
        return np.nan

    return float(t_contact)


def build_incoming_pitch_from_sensor(vx0, vy0, vz0, ax, ay, az, release_pos_y):
    t_contact = solve_contact_time_from_sensor(release_pos_y, vy0, ay)

    if not np.isfinite(t_contact):
        return {
            "t_contact": np.nan,
            "vin_x": np.nan,
            "vin_y": np.nan,
            "vin_z": np.nan,
        }

    vin_x = float(vx0) + float(ax) * t_contact
    vin_y = float(vy0) + float(ay) * t_contact
    vin_z = float(vz0) + float(az) * t_contact

    return {
        "t_contact": float(t_contact),
        "vin_x": vin_x,
        "vin_y": vin_y,
        "vin_z": vin_z,
    }


def relevant_omega_minus_from_sensor(
    release_spin_rate_rpm,
    spin_axis_deg,
    phi_rad,
    t_contact,
    asr=ASR_DEFAULT,
    k_torque=SPIN_TORQUE_K,
    spin_axis_offset_deg=SPIN_AXIS_OFFSET_DEG,
    sign_correction=SPIN_SIGN_CORRECTION,
):
    if not np.isfinite(t_contact):
        return np.nan

    # c-hat(phi) = (cos phi, -sin phi, 0)
    chat_x = np.cos(phi_rad)
    chat_y = -np.sin(phi_rad)
    chat_z = 0.0

    omega_release_rpm = float(release_spin_rate_rpm)
    omega_active_release_rpm = omega_release_rpm * float(asr)

    axis_rad = np.deg2rad(float(spin_axis_deg) + float(spin_axis_offset_deg))

    # Same proxy used in your preprocessing:
    # active spin axis approximated in the x-z plane
    omega_hat_x = np.cos(axis_rad)
    omega_hat_y = 0.0
    omega_hat_z = np.sin(axis_rad)

    # Nathan-style exponential spin-down
    decay_factor = np.exp(-0.020 * float(k_torque) * (t_contact * FPS_TO_MPH * 90.0))
    omega_contact_rpm = omega_active_release_rpm * decay_factor

    omega_vec_x_rpm = omega_contact_rpm * omega_hat_x
    omega_vec_y_rpm = omega_contact_rpm * omega_hat_y
    omega_vec_z_rpm = omega_contact_rpm * omega_hat_z

    omega_dot_chat_rpm = (
        omega_vec_x_rpm * chat_x +
        omega_vec_y_rpm * chat_y +
        omega_vec_z_rpm * chat_z
    )

    omega_minus_rpm = float(sign_correction) * (-omega_dot_chat_rpm)
    return float(omega_minus_rpm * RPM_TO_RAD_S)


# ---------------------------------------------------------
# Event cache: RAW observations only
# ---------------------------------------------------------

def prepare_event_cache(event_row):
    """
    Store only raw / near-raw observed inputs.
    Derived quantities are rebuilt downstream for each m*.
    """
    out = {
        "launch_speed_mph": float(event_row["launch_speed"]),
        "launch_angle_deg": float(event_row["launch_angle"]),
        "hc_x": float(event_row["hc_x"]),
        "hc_y": float(event_row["hc_y"]),
        "bat_speed_mph": float(event_row["bat_speed"]),
        "attack_angle_deg": float(event_row["attack_angle"]),
        "attack_direction_deg": float(event_row["attack_direction"]),
        "vx0": float(event_row["vx0"]),
        "vy0": float(event_row["vy0"]),
        "vz0": float(event_row["vz0"]),
        "ax": float(event_row["ax"]),
        "ay": float(event_row["ay"]),
        "az": float(event_row["az"]),
        "release_pos_y": float(event_row["release_pos_y"]),
        "release_spin_rate_rpm": float(event_row["release_spin_rate"]),
        "spin_axis_deg": float(event_row["spin_axis"]),
    }

    # Optional raw active-spin ratio, if you later add it
    if "active_spin_ratio_used" in event_row.index and pd.notna(event_row["active_spin_ratio_used"]):
        out["active_spin_ratio"] = float(event_row["active_spin_ratio_used"])
    else:
        out["active_spin_ratio"] = float(ASR_DEFAULT)

    return out


def default_sensor_sigmas():
    """
    Tight sensor-space Gaussian perturbations.
    Units match prepare_event_cache().
    """
    return {
        "launch_speed_mph": 0.25,
        "launch_angle_deg": 0.50,
        "hc_x": 0.75,
        "hc_y": 0.75,
        "bat_speed_mph": 0.25,
        "attack_angle_deg": 0.50,
        "attack_direction_deg": 0.75,
        "vx0": 0.25,
        "vy0": 0.25,
        "vz0": 0.25,
        "ax": 0.15,
        "ay": 0.15,
        "az": 0.15,
        "release_pos_y": 0.05,
        "release_spin_rate_rpm": 25.0,
        "spin_axis_deg": 1.0,
    }


def sample_measurement_star(ev_obs, rng, sensor_sigmas=None):
    """
    Sample m* directly in sensor space, then rebuild all derived quantities later.
    """
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    ev = dict(ev_obs)

    for k, sd in sensor_sigmas.items():
        ev[k] = float(rng.normal(ev_obs[k], sd))

    # Reasonable clipping / wrapping
    ev["launch_speed_mph"] = max(1.0, ev["launch_speed_mph"])
    ev["bat_speed_mph"] = max(1.0, ev["bat_speed_mph"])
    ev["launch_angle_deg"] = float(np.clip(ev["launch_angle_deg"], -90.0, 90.0))
    ev["attack_angle_deg"] = float(np.clip(ev["attack_angle_deg"], -90.0, 90.0))
    ev["attack_direction_deg"] = float(np.clip(ev["attack_direction_deg"], -180.0, 180.0))
    ev["release_spin_rate_rpm"] = max(1.0, ev["release_spin_rate_rpm"])
    ev["spin_axis_deg"] = wrap_deg_360(ev["spin_axis_deg"])

    return ev


def log_p_mstar_given_mobs(ev_star, ev_obs, sensor_sigmas=None):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    ll = 0.0
    for k, sd in sensor_sigmas.items():
        ll += norm.logpdf(ev_star[k], loc=ev_obs[k], scale=sd)
    return float(ll)


def rebuild_observed_event(ev_raw):
    """
    Convert one raw sensor-space draw m* into the 'observed/exogenous'
    event dictionary actually used by the inverse solver.
    """
    phi, hc_x_field, hc_y_field = spray_phi_from_hc(ev_raw["hc_x"], ev_raw["hc_y"])
    theta_abs = np.deg2rad(float(ev_raw["launch_angle_deg"]))
    s_obs = float(ev_raw["launch_speed_mph"]) * MPH_TO_FPS

    bat = build_bat_velocity_from_sensor(
        bat_speed_mph=ev_raw["bat_speed_mph"],
        attack_angle_deg=ev_raw["attack_angle_deg"],
        attack_direction_deg=ev_raw["attack_direction_deg"],
    )

    pitch = build_incoming_pitch_from_sensor(
        vx0=ev_raw["vx0"],
        vy0=ev_raw["vy0"],
        vz0=ev_raw["vz0"],
        ax=ev_raw["ax"],
        ay=ev_raw["ay"],
        az=ev_raw["az"],
        release_pos_y=ev_raw["release_pos_y"],
    )

    omega_minus = relevant_omega_minus_from_sensor(
        release_spin_rate_rpm=ev_raw["release_spin_rate_rpm"],
        spin_axis_deg=ev_raw["spin_axis_deg"],
        phi_rad=phi,
        t_contact=pitch["t_contact"],
        asr=ev_raw.get("active_spin_ratio", ASR_DEFAULT),
    )

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    vB_r_m = pitch["vin_x"] * sin_phi + pitch["vin_y"] * cos_phi
    vB_z_m = pitch["vin_z"]

    vb_r_m = bat["vbat_x"] * sin_phi + bat["vbat_y"] * cos_phi
    vb_z_m = bat["vbat_z"]

    dv_r = vb_r_m - vB_r_m
    dv_z = vb_z_m - vB_z_m

    bat_angle_proj = np.arctan2(vb_z_m, vb_r_m)
    theta_rel_bat = wrap_angle_pi(theta_abs - bat_angle_proj)

    return {
        # raw draw kept for bookkeeping
        "raw_star": dict(ev_raw),

        # rebuilt observables for solver
        "phi": float(phi),
        "theta": float(theta_abs),
        "theta_abs": float(theta_abs),
        "s_obs": float(s_obs),
        "omega_minus": float(omega_minus),

        "vB_r_m": float(vB_r_m),
        "vB_z_m": float(vB_z_m),
        "vb_r_m": float(vb_r_m),
        "vb_z_m": float(vb_z_m),
        "dv_r": float(dv_r),
        "dv_z": float(dv_z),

        "cos_theta": float(np.cos(theta_abs)),
        "sin_theta": float(np.sin(theta_abs)),
        "bat_angle_proj": float(bat_angle_proj),
        "theta_rel_bat": float(theta_rel_bat),

        # extra diagnostics / provenance
        "hc_x_field": float(hc_x_field),
        "hc_y_field": float(hc_y_field),
        "t_contact": float(pitch["t_contact"]),
        "vin_x": float(pitch["vin_x"]),
        "vin_y": float(pitch["vin_y"]),
        "vin_z": float(pitch["vin_z"]),
        "vbat_x": float(bat["vbat_x"]),
        "vbat_y": float(bat["vbat_y"]),
        "vbat_z": float(bat["vbat_z"]),
    }


# ---------------------------------------------------------
# Exact x -> psi mapping
# ---------------------------------------------------------

def fn_linear_coeffs_for_x(x_in, hitter, ev):
    e_y_val = e_y_player(x_in, hitter)
    r_y_val = float(r_y_of_x(
        x_in,
        hitter.x_cm,
        hitter.m_ball_oz,
        hitter.bat_weight_oz,
        hitter.I0_oz_in2,
    ))
    coeff = (1.0 + e_y_val) / (1.0 + r_y_val)

    A = ev["vB_r_m"] + coeff * ev["dv_r"] - ev["s_obs"] * ev["cos_theta"]
    B = ev["vB_z_m"] + coeff * ev["dv_z"] - ev["s_obs"] * ev["sin_theta"]
    return float(A), float(B), float(e_y_val), float(r_y_val), float(coeff)


def analytic_psi_abs_root_for_x(
    x_in,
    hitter,
    ev,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    root_tol=DEFAULT_ROOT_TOL,
):
    """
    On support width < pi, there is at most one admissible root.
    """
    A, B, _, _, _ = fn_linear_coeffs_for_x(x_in, hitter, ev)
    amp = np.hypot(A, B)
    if amp < root_tol:
        return None

    psi0 = wrap_angle_pi(np.arctan2(-A, B))

    lo = np.deg2rad(psi_abs_min_deg)
    hi = np.deg2rad(psi_abs_max_deg)

    if lo <= psi0 <= hi:
        return float(psi0)
    return None


def psi_rel_from_nt(vb_n_m, vb_t_m):
    return float(np.arctan2(vb_t_m, vb_n_m))


def D_of_psi_rel(R_loc_in, r_ball_in, psi_rel):
    return float((R_loc_in + r_ball_in) * np.sin(psi_rel))


def decompose_event_at_psi_abs(psi_abs, ev):
    cp = np.cos(psi_abs)
    sp = np.sin(psi_abs)

    cos_theta_minus_psi = ev["cos_theta"] * cp + ev["sin_theta"] * sp
    sin_theta_minus_psi = ev["sin_theta"] * cp - ev["cos_theta"] * sp

    vB_n_m = ev["vB_r_m"] * cp + ev["vB_z_m"] * sp
    vB_t_m = -ev["vB_r_m"] * sp + ev["vB_z_m"] * cp

    vb_n_m = ev["vb_r_m"] * cp + ev["vb_z_m"] * sp
    vb_t_m = -ev["vb_r_m"] * sp + ev["vb_z_m"] * cp

    d_n = ev["dv_r"] * cp + ev["dv_z"] * sp
    d_t = -ev["dv_r"] * sp + ev["dv_z"] * cp

    psi_rel = psi_rel_from_nt(vb_n_m, vb_t_m)
    psi_regime_abs = abs(psi_rel)

    Vn_plus_obs = ev["s_obs"] * cos_theta_minus_psi
    Vt_plus_obs = ev["s_obs"] * sin_theta_minus_psi

    return {
        "cp": float(cp),
        "sp": float(sp),
        "cos_theta_minus_psi": float(cos_theta_minus_psi),
        "sin_theta_minus_psi": float(sin_theta_minus_psi),
        "vB_n_m": float(vB_n_m),
        "vB_t_m": float(vB_t_m),
        "vb_n_m": float(vb_n_m),
        "vb_t_m": float(vb_t_m),
        "d_n": float(d_n),
        "d_t": float(d_t),
        "psi_rel": float(psi_rel),
        "psi_regime_abs": float(psi_regime_abs),
        "Vn_plus_obs": float(Vn_plus_obs),
        "Vt_plus_obs": float(Vt_plus_obs),
    }


def classify_regime_from_psi_rel(psi_rel_abs, mu, g2, e_y):
    tanv = float(np.tan(psi_rel_abs))
    lower = float(mu * g2)
    upper = float(3.5 * mu * (1.0 + e_y))

    if tanv < lower:
        regime = "stick-slip"
    elif tanv < upper:
        regime = "slip-stick-slip"
    else:
        regime = "gross-slip"

    return {
        "regime_label": regime,
        "tan_psi_regime": tanv,
        "regime_lower": lower,
        "regime_upper": upper,
    }


def log_prior_ex_from_regime(e_x_val, regime_label):
    """
    Hard regime admissibility:
      - gross-slip          => inadmissible
      - stick-slip          => N(0.4, 0.1)
      - slip-stick-slip     => Uniform(0, 0.6)
    """
    if not np.isfinite(e_x_val):
        return -np.inf

    if regime_label == "gross-slip":
        return -np.inf

    if regime_label == "stick-slip":
        return float(norm.logpdf(e_x_val, loc=0.4, scale=0.1))

    if regime_label == "slip-stick-slip":
        if 0.0 <= e_x_val <= 0.6:
            return float(-np.log(0.6))
        return -np.inf

    return -np.inf


def solve_state_from_x(
    x_in,
    hitter,
    ev,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    denom_tol=DEFAULT_DENOM_TOL,
):
    psi_abs = analytic_psi_abs_root_for_x(
        x_in=x_in,
        hitter=hitter,
        ev=ev,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
    )
    if psi_abs is None:
        return None

    dec = decompose_event_at_psi_abs(psi_abs, ev)

    R_loc_in = R_player(x_in, hitter)
    e_y_val = e_y_player(x_in, hitter)

    r_y_val = float(r_y_of_x(
        x_in,
        hitter.x_cm,
        hitter.m_ball_oz,
        hitter.bat_weight_oz,
        hitter.I0_oz_in2,
    ))

    r_x_val = float(r_x_of_x(
        x_in,
        hitter.x_cm,
        hitter.m_ball_oz,
        hitter.alpha,
        hitter.bat_weight_oz,
        hitter.I0_oz_in2,
        R_loc_in,
        hitter.Iz_oz_in2,
    ))

    r_ball_in = float(hitter.r_ball_in)
    r_ball_ft = r_ball_in / 12.0
    omega_minus = float(ev["omega_minus"])

    denom = hitter.alpha * (dec["d_t"] - r_ball_ft * omega_minus)
    if abs(denom) < denom_tol:
        return None

    numer_ex = (
        (dec["Vt_plus_obs"] - dec["vB_t_m"])
        * (1.0 + r_x_val)
        * (1.0 + hitter.alpha)
    )
    e_x_val = numer_ex / denom - 1.0

    D_in = D_of_psi_rel(R_loc_in, r_ball_in, dec["psi_rel"])

    numer_w = (
        (dec["vB_t_m"] - dec["Vt_plus_obs"])
        - (D_in / r_ball_in) * dec["Vn_plus_obs"]
    )
    omega_plus_val = omega_minus + numer_w / (hitter.alpha * r_ball_ft)

    return {
        **dec,
        "psi_abs": float(psi_abs),
        "R_loc_in": float(R_loc_in),
        "D_in": float(D_in),
        "e_y_val": float(e_y_val),
        "r_y_val": float(r_y_val),
        "r_x_val": float(r_x_val),
        "e_x_val": float(e_x_val),
        "omega_plus_val": float(omega_plus_val),
    }


# ---------------------------------------------------------
# Residuals
# ---------------------------------------------------------

def F_n_scalar_from_ev(psi_abs, x_in, hitter, ev):
    cp = np.cos(psi_abs)
    sp = np.sin(psi_abs)

    e_y_val = e_y_player(x_in, hitter)
    r_y_val = float(r_y_of_x(
        x_in,
        hitter.x_cm,
        hitter.m_ball_oz,
        hitter.bat_weight_oz,
        hitter.I0_oz_in2,
    ))
    coeff = (1.0 + e_y_val) / (1.0 + r_y_val)

    vB_n = ev["vB_r_m"] * cp + ev["vB_z_m"] * sp
    d_n = ev["dv_r"] * cp + ev["dv_z"] * sp
    cos_theta_minus_psi = ev["cos_theta"] * cp + ev["sin_theta"] * sp

    return float(vB_n + coeff * d_n - ev["s_obs"] * cos_theta_minus_psi)


def F_t_exact_from_ev(psi_abs, x_in, e_x_val, hitter, ev, r_x_val):
    dec = decompose_event_at_psi_abs(psi_abs, ev)
    r_ball_ft = float(hitter.r_ball_in) / 12.0

    return float(
        dec["vB_t_m"]
        + hitter.alpha * (1.0 + e_x_val)
          / ((1.0 + r_x_val) * (1.0 + hitter.alpha))
          * (dec["d_t"] - r_ball_ft * ev["omega_minus"])
        - dec["Vt_plus_obs"]
    )


def F_w_exact_from_ev(psi_abs, omega_plus_val, hitter, ev, D_in):
    dec = decompose_event_at_psi_abs(psi_abs, ev)
    r_ball_in = float(hitter.r_ball_in)
    r_ball_ft = r_ball_in / 12.0

    return float(
        hitter.alpha * r_ball_ft * (omega_plus_val - ev["omega_minus"])
        - (dec["vB_t_m"] - dec["Vt_plus_obs"])
        + (D_in / r_ball_in) * dec["Vn_plus_obs"]
    )


# ---------------------------------------------------------
# Single-state evaluation on the exact 1D manifold
# ---------------------------------------------------------

def evaluate_x_state(
    x_in,
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    enforce_positive_normal=True,
    enforce_normal_impulse_dominance=True,
    denom_tol=DEFAULT_DENOM_TOL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
):
    state = solve_state_from_x(
        x_in=x_in,
        hitter=hitter,
        ev=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        denom_tol=denom_tol,
    )
    if state is None:
        return None

    Vn_plus = state["Vn_plus_obs"]
    Vt_plus = state["Vt_plus_obs"]

    if enforce_positive_normal and not (Vn_plus > 0.0):
        return None

    vt_over_vn_abs = np.inf if abs(Vn_plus) < denom_tol else abs(Vt_plus) / abs(Vn_plus)
    if enforce_normal_impulse_dominance and not (vt_over_vn_abs < 1.0):
        return None

    if not (state["d_n"] > 0.0):
        return None

    if not np.isfinite(state["D_in"]):
        return None
    if abs(state["D_in"]) > (state["R_loc_in"] + hitter.r_ball_in + 1e-8):
        return None

    regime_info = classify_regime_from_psi_rel(
        psi_rel_abs=state["psi_regime_abs"],
        mu=mu,
        g2=g2,
        e_y=state["e_y_val"],
    )

    log_prior_ex = log_prior_ex_from_regime(
        e_x_val=state["e_x_val"],
        regime_label=regime_info["regime_label"],
    )
    if not np.isfinite(log_prior_ex):
        return None

    Fn_res = F_n_scalar_from_ev(state["psi_abs"], x_in, hitter, ev_star)
    Ft_res = F_t_exact_from_ev(
        state["psi_abs"], x_in, state["e_x_val"], hitter, ev_star, state["r_x_val"]
    )
    Fw_res = F_w_exact_from_ev(
        state["psi_abs"], state["omega_plus_val"], hitter, ev_star, state["D_in"]
    )

    R_prof = (
        (Fn_res / sigma_n) ** 2
        + (Ft_res / sigma_t) ** 2
        + (Fw_res / sigma_w) ** 2
    )

    log_target = float(log_prior_ex - 0.5 * R_prof)

    return {
        "x_in": float(x_in),
        "psi_abs": float(state["psi_abs"]),
        "psi_rel": float(state["psi_rel"]),
        "psi_regime_abs": float(state["psi_regime_abs"]),
        "theta_abs": float(ev_star["theta_abs"]),
        "s_obs": float(ev_star["s_obs"]),
        "vB_r_m": float(ev_star["vB_r_m"]),
        "vB_z_m": float(ev_star["vB_z_m"]),
        "vb_r_m": float(ev_star["vb_r_m"]),
        "vb_z_m": float(ev_star["vb_z_m"]),
        "e_y": float(state["e_y_val"]),
        "e_x": float(state["e_x_val"]),
        "omega_minus": float(ev_star["omega_minus"]),
        "omega_plus": float(state["omega_plus_val"]),
        "R_in": float(state["R_loc_in"]),
        "D_in": float(state["D_in"]),
        "r_y": float(state["r_y_val"]),
        "r_x": float(state["r_x_val"]),
        "Vn_plus_obs": float(state["Vn_plus_obs"]),
        "Vt_plus_obs": float(state["Vt_plus_obs"]),
        "vt_over_vn_abs": float(vt_over_vn_abs),
        "Fn_res": float(Fn_res),
        "Ft_res": float(Ft_res),
        "Fw_res": float(Fw_res),
        "R_prof": float(R_prof),
        "log_prior_ex": float(log_prior_ex),
        "log_target": float(log_target),
        "regime_label": regime_info["regime_label"],
        "tan_psi_regime": float(regime_info["tan_psi_regime"]),
        "regime_lower": float(regime_info["regime_lower"]),
        "regime_upper": float(regime_info["regime_upper"]),
        "theta_rel_bat": float(ev_star["theta_rel_bat"]),
        "bat_angle_proj": float(ev_star["bat_angle_proj"]),
        "ev_star": ev_star,
    }


# ---------------------------------------------------------
# Admissible x-interval detection
# ---------------------------------------------------------

def _contiguous_true_runs(mask):
    runs = []
    in_run = False
    start = None
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run = True
            start = i
        elif (not m) and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def find_admissible_x_intervals(
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    n_x_grid=401,
):
    x_lo = hitter.bat_length - 11.0
    x_hi = hitter.bat_length

    x_grid = np.linspace(x_lo, x_hi, n_x_grid)
    states = []
    admissible = []

    for x_in in x_grid:
        st = evaluate_x_state(
            x_in=x_in,
            hitter=hitter,
            ev_star=ev_star,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )
        states.append(st)
        admissible.append(st is not None)

    admissible = np.asarray(admissible, dtype=bool)
    runs = _contiguous_true_runs(admissible)

    intervals = []
    for i0, i1 in runs:
        left = float(x_grid[i0])
        right = float(x_grid[i1])
        if right > left:
            intervals.append((left, right))

    return {
        "x_grid": x_grid,
        "states": states,
        "admissible_mask": admissible,
        "intervals": intervals,
    }

# =========================================================
# FAST MANIFOLD MCMC BLOCK REPLACEMENT
#
# Main speedups:
#   - vectorized analytic manifold evaluation x -> (psi, e_x, omega_plus)
#   - coarse x scan only to locate admissible runs
#   - boundary refinement by bisection on admissibility
#   - fine precomputation ONLY on admissible intervals
#   - direct conditional resampling of x from precomputed 1-D manifold map
#   - raw sensor-space m* update retained
#   - tqdm progress / ETA
#   - ArviZ InferenceData output for PyMC-style diagnostics / plotting
#
# Assumes the replacement FUNCTION block is already in memory, including:
#   - prepare_event_cache
#   - default_sensor_sigmas
#   - sample_measurement_star
#   - rebuild_observed_event
#   - evaluate_x_state
#   - log_p_mstar_given_mobs
#   - _contiguous_true_runs
#   - constants like DEFAULT_MU, DEFAULT_G2_BASEBALL, DEFAULT_SIGMA_*
#   - e_eff, Hitter.R / hitter.R_at_x
# =========================================================

import time
import numpy as np
import pandas as pd
import arviz as az
from tqdm.auto import tqdm


# ---------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------

def _normal_logpdf_vec(x, mu, sd):
    z = (x - mu) / sd
    return -0.5 * z * z - np.log(sd) - 0.5 * np.log(2.0 * np.pi)


def _wrap_angle_pi_vec(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _safe_exp_normalized(logw):
    m = np.max(logw)
    if not np.isfinite(m):
        return np.zeros_like(logw)
    return np.exp(logw - m)


def _midpoint_cell_widths(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 1:
        return np.array([1.0], dtype=float)
    w = np.empty(n, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if n > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _summarize_draws(df):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for col in ["x_in", "psi_abs_deg", "psi_rel_deg", "e_x", "omega_plus_rad_s"]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
        rows.append({
            "parameter": col,
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(q50),
            "q05": float(q05),
            "q95": float(q95),
        })
    return pd.DataFrame(rows)


def _df_to_idata(df, meta=None):
    if df.empty:
        return az.from_dict(posterior={})

    posterior_cols = [
        "x_in", "psi_abs_rad", "psi_abs_deg", "psi_rel_rad", "psi_rel_deg",
        "e_x", "e_y", "omega_minus_rad_s", "omega_plus_rad_s",
        "D_in", "R_in", "r_x", "r_y",
        "Fn_res", "Ft_res", "Fw_res", "R_prof",
        "phi_rad", "phi_deg", "theta_abs_rad", "theta_abs_deg",
        "s_obs_fps", "vB_r_m_fps", "vB_z_m_fps", "vb_r_m_fps", "vb_z_m_fps",
    ]
    posterior = {
        c: df[c].to_numpy(dtype=float)[None, :]
        for c in posterior_cols if c in df.columns
    }

    sample_stats_cols = [
        "log_target_conditional", "log_p_mstar_given_mobs", "log_joint_proxy",
        "mstar_accepted", "direct_x_resample_success", "map_rebuilt",
        "map_total_points", "map_interval_count", "map_total_width",
    ]
    sample_stats = {
        c: df[c].to_numpy(dtype=float)[None, :]
        for c in sample_stats_cols if c in df.columns
    }

    idata = az.from_dict(
        posterior=posterior,
        sample_stats=sample_stats,
    )

    if meta is not None:
        idata.attrs.update({k: str(v) for k, v in meta.items()})

    return idata


# ---------------------------------------------------------
# Vectorized analytic manifold evaluation
# ---------------------------------------------------------

def vectorized_manifold_eval(
    x_arr,
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    denom_tol=DEFAULT_DENOM_TOL,
):
    """
    Evaluate the exact 1-D manifold analytically and vectorized over x.
    For fixed m*, psi(x), e_x(x), omega_plus(x) are deterministic.
    """
    x_arr = np.asarray(x_arr, dtype=float)

    # hitter / physical constants
    L = float(hitter.bat_length)
    x_cm = float(hitter.x_cm)
    m_oz = float(hitter.m_ball_oz)
    M_oz = float(hitter.bat_weight_oz)
    I0 = float(hitter.I0_oz_in2)
    Iz = float(hitter.Iz_oz_in2)
    alpha = float(hitter.alpha)
    r_ball_in = float(hitter.r_ball_in)
    r_ball_ft = r_ball_in / 12.0

    # rebuilt event quantities
    s_obs = float(ev_star["s_obs"])
    omega_minus = float(ev_star["omega_minus"])
    vB_r_m = float(ev_star["vB_r_m"])
    vB_z_m = float(ev_star["vB_z_m"])
    vb_r_m = float(ev_star["vb_r_m"])
    vb_z_m = float(ev_star["vb_z_m"])
    dv_r = float(ev_star["dv_r"])
    dv_z = float(ev_star["dv_z"])
    cos_theta = float(ev_star["cos_theta"])
    sin_theta = float(ev_star["sin_theta"])

    # bat profiles / recoil factors
    e_y = e_eff(x_arr, L=L, clip=True)
    R_in = Hitter.R(x_arr, L=L, clip=True)
    b = x_arr - x_cm

    r_y = m_oz * (1.0 / M_oz + (b ** 2) / I0)
    r_x = (m_oz * alpha / (1.0 + alpha)) * (
        1.0 / M_oz + (b ** 2) / I0 + (R_in ** 2) / Iz
    )

    coeff = (1.0 + e_y) / (1.0 + r_y)

    # exact analytic root psi(x)
    A = vB_r_m + coeff * dv_r - s_obs * cos_theta
    B = vB_z_m + coeff * dv_z - s_obs * sin_theta
    amp = np.hypot(A, B)

    psi_abs = _wrap_angle_pi_vec(np.arctan2(-A, B))
    psi_lo = np.deg2rad(float(psi_abs_min_deg))
    psi_hi = np.deg2rad(float(psi_abs_max_deg))
    psi_on_support = (amp >= DEFAULT_ROOT_TOL) & (psi_abs >= psi_lo) & (psi_abs <= psi_hi)

    cp = np.cos(psi_abs)
    sp = np.sin(psi_abs)

    cos_theta_minus_psi = cos_theta * cp + sin_theta * sp
    sin_theta_minus_psi = sin_theta * cp - cos_theta * sp

    vB_n_m = vB_r_m * cp + vB_z_m * sp
    vB_t_m = -vB_r_m * sp + vB_z_m * cp

    vb_n_m = vb_r_m * cp + vb_z_m * sp
    vb_t_m = -vb_r_m * sp + vb_z_m * cp

    d_n = dv_r * cp + dv_z * sp
    d_t = -dv_r * sp + dv_z * cp

    psi_rel = np.arctan2(vb_t_m, vb_n_m)
    psi_regime_abs = np.abs(psi_rel)

    Vn_plus_obs = s_obs * cos_theta_minus_psi
    Vt_plus_obs = s_obs * sin_theta_minus_psi

    denom = alpha * (d_t - r_ball_ft * omega_minus)
    denom_ok = np.abs(denom) >= denom_tol

    e_x = np.full_like(x_arr, np.nan, dtype=float)
    numer_ex = ((Vt_plus_obs - vB_t_m) * (1.0 + r_x) * (1.0 + alpha))
    e_x[denom_ok] = numer_ex[denom_ok] / denom[denom_ok] - 1.0

    D_in = (R_in + r_ball_in) * np.sin(psi_rel)

    omega_plus = omega_minus + (
        ((vB_t_m - Vt_plus_obs) - (D_in / r_ball_in) * Vn_plus_obs)
        / (alpha * r_ball_ft)
    )

    # regime classification
    tan_psi_regime = np.tan(psi_regime_abs)
    regime_lower = mu * g2
    regime_upper = 3.5 * mu * (1.0 + e_y)

    regime_code = np.full_like(x_arr, 2, dtype=int)  # 0 stick-slip, 1 slip-stick-slip, 2 gross-slip
    regime_code[tan_psi_regime < regime_lower] = 0
    regime_code[(tan_psi_regime >= regime_lower) & (tan_psi_regime < regime_upper)] = 1

    # regime-dependent prior on e_x
    log_prior_ex = np.full_like(x_arr, -np.inf, dtype=float)

    # stick-slip => N(0.4, 0.1)
    m0 = regime_code == 0
    log_prior_ex[m0] = _normal_logpdf_vec(e_x[m0], 0.4, 0.1)

    # slip-stick-slip => Uniform(0, 0.6)
    m1 = regime_code == 1
    in_u = m1 & (e_x >= 0.0) & (e_x <= 0.6)
    log_prior_ex[in_u] = -np.log(0.6)

    # residuals (should be ~0 on exact manifold up to numerics)
    Fn_res = vB_n_m + coeff * d_n - s_obs * cos_theta_minus_psi

    Ft_res = (
        vB_t_m
        + alpha * (1.0 + e_x) / ((1.0 + r_x) * (1.0 + alpha))
          * (d_t - r_ball_ft * omega_minus)
        - Vt_plus_obs
    )

    Fw_res = (
        alpha * r_ball_ft * (omega_plus - omega_minus)
        - (vB_t_m - Vt_plus_obs)
        + (D_in / r_ball_in) * Vn_plus_obs
    )

    R_prof = (
        (Fn_res / sigma_n) ** 2
        + (Ft_res / sigma_t) ** 2
        + (Fw_res / sigma_w) ** 2
    )

    log_target = log_prior_ex - 0.5 * R_prof

    vt_over_vn_abs = np.where(np.abs(Vn_plus_obs) >= denom_tol, np.abs(Vt_plus_obs) / np.abs(Vn_plus_obs), np.inf)

    admissible = (
        psi_on_support
        & np.isfinite(e_x)
        & np.isfinite(omega_plus)
        & np.isfinite(D_in)
        & (Vn_plus_obs > 0.0)
        & (vt_over_vn_abs < 1.0)
        & (d_n > 0.0)
        & (np.abs(D_in) <= (R_in + r_ball_in + 1e-8))
        & np.isfinite(log_target)
        & (regime_code != 2)  # gross-slip inadmissible
    )

    regime_label = np.full_like(x_arr, "gross-slip", dtype=object)
    regime_label[regime_code == 0] = "stick-slip"
    regime_label[regime_code == 1] = "slip-stick-slip"

    return {
        "x": x_arr,
        "psi_abs": psi_abs,
        "psi_rel": psi_rel,
        "psi_regime_abs": psi_regime_abs,
        "e_x": e_x,
        "e_y": e_y,
        "omega_plus": omega_plus,
        "omega_minus": np.full_like(x_arr, omega_minus, dtype=float),
        "D_in": D_in,
        "R_in": R_in,
        "r_x": r_x,
        "r_y": r_y,
        "Vn_plus_obs": Vn_plus_obs,
        "Vt_plus_obs": Vt_plus_obs,
        "Fn_res": Fn_res,
        "Ft_res": Ft_res,
        "Fw_res": Fw_res,
        "R_prof": R_prof,
        "log_prior_ex": log_prior_ex,
        "log_target": log_target,
        "vt_over_vn_abs": vt_over_vn_abs,
        "tan_psi_regime": tan_psi_regime,
        "regime_lower": np.full_like(x_arr, regime_lower, dtype=float),
        "regime_upper": regime_upper,
        "regime_code": regime_code,
        "regime_label": regime_label,
        "admissible": admissible,
    }


# ---------------------------------------------------------
# Boundary refinement on admissibility predicate
# ---------------------------------------------------------

def refine_interval_edge(
    x_bad,
    x_good,
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    max_iter=30,
):
    """
    Bisection between an inadmissible x and admissible x to refine the edge.
    Returns the admissible-side limit.
    """
    lo = float(x_bad)
    hi = float(x_good)

    st_hi = evaluate_x_state(
        x_in=hi,
        hitter=hitter,
        ev_star=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
    )
    if st_hi is None:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        st_mid = evaluate_x_state(
            x_in=mid,
            hitter=hitter,
            ev_star=ev_star,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )
        if st_mid is None:
            lo = mid
        else:
            hi = mid

    return float(hi)


# ---------------------------------------------------------
# Build fast precomputed 1-D manifold map for one m*
# ---------------------------------------------------------

def build_fast_manifold_map(
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    coarse_n=81,
    fine_points_per_interval=96,
):
    """
    Strategy:
      1) cheap coarse vectorized scan over full support
      2) detect 1-2 admissible runs
      3) refine boundaries by scalar bisection
      4) vectorized fine map only on refined admissible intervals
    """
    x_lo = float(hitter.bat_length - 11.0)
    x_hi = float(hitter.bat_length)

    # coarse scan
    x_coarse = np.linspace(x_lo, x_hi, coarse_n)
    coarse = vectorized_manifold_eval(
        x_coarse,
        hitter=hitter,
        ev_star=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
    )

    mask = coarse["admissible"]
    runs = _contiguous_true_runs(mask)

    if not runs:
        return {
            "intervals": [],
            "x": np.array([], dtype=float),
            "admissible_mask": np.array([], dtype=bool),
            "log_target": np.array([], dtype=float),
            "map": None,
            "total_width": 0.0,
            "total_points": 0,
        }

    refined_intervals = []
    for i0, i1 in runs:
        left_good = x_coarse[i0]
        right_good = x_coarse[i1]

        # left edge
        if i0 == 0:
            left_ref = float(left_good)
        else:
            left_bad = x_coarse[i0 - 1]
            left_ref = refine_interval_edge(
                x_bad=left_bad,
                x_good=left_good,
                hitter=hitter,
                ev_star=ev_star,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
            )

        # right edge
        if i1 == len(x_coarse) - 1:
            right_ref = float(right_good)
        else:
            right_bad = x_coarse[i1 + 1]
            # reverse roles so "good" is on the left and "bad" on the right
            # refine to admissible-side limit from the right
            lo = float(right_good)
            hi = float(right_bad)
            st_lo = evaluate_x_state(
                x_in=lo,
                hitter=hitter,
                ev_star=ev_star,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
            )
            if st_lo is None:
                right_ref = lo
            else:
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    st_mid = evaluate_x_state(
                        x_in=mid,
                        hitter=hitter,
                        ev_star=ev_star,
                        psi_abs_min_deg=psi_abs_min_deg,
                        psi_abs_max_deg=psi_abs_max_deg,
                        mu=mu,
                        g2=g2,
                        sigma_n=sigma_n,
                        sigma_t=sigma_t,
                        sigma_w=sigma_w,
                    )
                    if st_mid is None:
                        hi = mid
                    else:
                        lo = mid
                right_ref = float(lo)

        if right_ref > left_ref:
            refined_intervals.append((left_ref, right_ref))

    if not refined_intervals:
        return {
            "intervals": [],
            "x": np.array([], dtype=float),
            "admissible_mask": np.array([], dtype=bool),
            "log_target": np.array([], dtype=float),
            "map": None,
            "total_width": 0.0,
            "total_points": 0,
        }

    # fine vectorized map only on refined intervals
    x_chunks = []
    interval_id = []

    for k, (a, b) in enumerate(refined_intervals):
        n_k = max(16, fine_points_per_interval)
        xk = np.linspace(a, b, n_k)
        x_chunks.append(xk)
        interval_id.extend([k] * len(xk))

    x_fine = np.concatenate(x_chunks)
    interval_id = np.asarray(interval_id, dtype=int)

    fine = vectorized_manifold_eval(
        x_fine,
        hitter=hitter,
        ev_star=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
    )

    total_width = float(sum(b - a for a, b in refined_intervals))

    return {
        "intervals": refined_intervals,
        "x": x_fine,
        "interval_id": interval_id,
        "admissible_mask": fine["admissible"],
        "log_target": fine["log_target"],
        "map": fine,
        "total_width": total_width,
        "total_points": int(len(x_fine)),
    }


# ---------------------------------------------------------
# Direct conditional resampling of x from precomputed manifold
# ---------------------------------------------------------

def sample_x_from_fast_manifold_map(
    manifold_map,
    hitter,
    ev_star,
    rng,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    max_tries=30,
):
    """
    Approximate direct conditional draw:
      - weight fine manifold points by exp(log_target) * local cell width
      - choose a point
      - jitter locally within the chosen cell / interval
      - validate with exact scalar evaluate_x_state
    """
    if manifold_map["map"] is None:
        return None, None, {
            "direct_x_resample_success": 0,
            "map_interval_count": 0,
            "map_total_width": 0.0,
            "map_total_points": 0,
        }

    x = manifold_map["x"]
    m = manifold_map["map"]
    valid = manifold_map["admissible_mask"]

    if not np.any(valid):
        return None, None, {
            "direct_x_resample_success": 0,
            "map_interval_count": len(manifold_map["intervals"]),
            "map_total_width": manifold_map["total_width"],
            "map_total_points": manifold_map["total_points"],
        }

    x_valid = x[valid]
    lt_valid = m["log_target"][valid]
    iid_valid = manifold_map["interval_id"][valid]

    cell_w = _midpoint_cell_widths(x_valid)
    w = _safe_exp_normalized(lt_valid) * cell_w
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return None, None, {
            "direct_x_resample_success": 0,
            "map_interval_count": len(manifold_map["intervals"]),
            "map_total_width": manifold_map["total_width"],
            "map_total_points": manifold_map["total_points"],
        }

    p = w / w_sum

    for _ in range(max_tries):
        j = int(rng.choice(len(x_valid), p=p))
        x0 = float(x_valid[j])
        iid = int(iid_valid[j])

        # local jitter bounds inside same interval
        interval_a, interval_b = manifold_map["intervals"][iid]

        if j == 0 or iid_valid[j - 1] != iid:
            left = interval_a
        else:
            left = 0.5 * (x_valid[j - 1] + x_valid[j])

        if j == len(x_valid) - 1 or iid_valid[j + 1] != iid:
            right = interval_b
        else:
            right = 0.5 * (x_valid[j] + x_valid[j + 1])

        x_prop = float(rng.uniform(left, right))

        st = evaluate_x_state(
            x_in=x_prop,
            hitter=hitter,
            ev_star=ev_star,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )
        if st is not None:
            return x_prop, st, {
                "direct_x_resample_success": 1,
                "map_interval_count": len(manifold_map["intervals"]),
                "map_total_width": manifold_map["total_width"],
                "map_total_points": manifold_map["total_points"],
            }

    # fallback to exact evaluation at selected grid point
    j = int(rng.choice(len(x_valid), p=p))
    x_prop = float(x_valid[j])
    st = evaluate_x_state(
        x_in=x_prop,
        hitter=hitter,
        ev_star=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
    )

    return x_prop, st, {
        "direct_x_resample_success": int(st is not None),
        "map_interval_count": len(manifold_map["intervals"]),
        "map_total_width": manifold_map["total_width"],
        "map_total_points": manifold_map["total_points"],
    }


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def initialize_fast_chain(
    event_row,
    hitter,
    rng,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    coarse_n=81,
    fine_points_per_interval=96,
    max_init_tries=300,
):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    ev_obs_raw = prepare_event_cache(event_row)

    for init_try in range(1, max_init_tries + 1):
        raw_star = sample_measurement_star(ev_obs_raw, rng, sensor_sigmas=sensor_sigmas)
        ev_star = rebuild_observed_event(raw_star)

        core_vals = [
            ev_star["phi"], ev_star["theta"], ev_star["s_obs"],
            ev_star["omega_minus"],
            ev_star["vB_r_m"], ev_star["vB_z_m"],
            ev_star["vb_r_m"], ev_star["vb_z_m"],
        ]
        if not np.all(np.isfinite(core_vals)):
            continue

        manifold_map = build_fast_manifold_map(
            hitter=hitter,
            ev_star=ev_star,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
            coarse_n=coarse_n,
            fine_points_per_interval=fine_points_per_interval,
        )
        if not manifold_map["intervals"]:
            continue

        x0, st0, _ = sample_x_from_fast_manifold_map(
            manifold_map=manifold_map,
            hitter=hitter,
            ev_star=ev_star,
            rng=rng,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )
        if st0 is not None:
            return {
                "ev_obs_raw": ev_obs_raw,
                "raw_star": raw_star,
                "ev_star": ev_star,
                "x_in": float(x0),
                "state": st0,
                "manifold_map": manifold_map,
                "init_try": init_try,
            }

    raise RuntimeError("Failed to initialize fast manifold chain with an admissible (m*, x) state.")


# ---------------------------------------------------------
# Single-event fast manifold sampler
# ---------------------------------------------------------

def run_single_event_fast_manifold_chain(
    event_row,
    hitter,
    n_iter=2000,
    burn=500,
    thin=2,
    seed=123,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=0.1,
    sigma_t=0.1,
    sigma_w=0.1,
    coarse_n=81,
    fine_points_per_interval=96,
    show_progress=True,
):
    """
    Blocked sampler:
      A) m* update by independent MH in raw sensor space
      B) x update by direct conditional resampling from a precomputed 1-D manifold map

    This is much faster than rescanning x every iteration.
    """
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    rng = np.random.default_rng(seed)

    init = initialize_fast_chain(
        event_row=event_row,
        hitter=hitter,
        rng=rng,
        sensor_sigmas=sensor_sigmas,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
        coarse_n=coarse_n,
        fine_points_per_interval=fine_points_per_interval,
    )

    ev_obs_raw = init["ev_obs_raw"]
    current_raw_star = init["raw_star"]
    current_ev_star = init["ev_star"]
    current_x = init["x_in"]
    current_state = init["state"]
    current_map = init["manifold_map"]

    draws = []
    accept_mstar = 0
    valid_mstar_prop = 0

    iterator = range(n_iter)
    if show_progress:
        iterator = tqdm(iterator, desc="fast manifold chain", unit="iter")

    for it in iterator:
        map_rebuilt = 0

        # ---------------------------------------------
        # Block A: propose raw m* directly in sensor space
        # ---------------------------------------------
        prop_raw_star = sample_measurement_star(ev_obs_raw, rng, sensor_sigmas=sensor_sigmas)
        prop_ev_star = rebuild_observed_event(prop_raw_star)

        prop_ok = np.all(np.isfinite([
            prop_ev_star["phi"], prop_ev_star["theta"], prop_ev_star["s_obs"],
            prop_ev_star["omega_minus"],
            prop_ev_star["vB_r_m"], prop_ev_star["vB_z_m"],
            prop_ev_star["vb_r_m"], prop_ev_star["vb_z_m"],
        ]))

        mstar_accepted = 0
        if prop_ok:
            prop_state_at_current_x = evaluate_x_state(
                x_in=current_x,
                hitter=hitter,
                ev_star=prop_ev_star,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
            )

            if prop_state_at_current_x is not None:
                valid_mstar_prop += 1
                log_alpha = float(prop_state_at_current_x["log_target"] - current_state["log_target"])

                if np.log(rng.uniform()) < log_alpha:
                    current_raw_star = prop_raw_star
                    current_ev_star = prop_ev_star
                    current_state = prop_state_at_current_x

                    current_map = build_fast_manifold_map(
                        hitter=hitter,
                        ev_star=current_ev_star,
                        psi_abs_min_deg=psi_abs_min_deg,
                        psi_abs_max_deg=psi_abs_max_deg,
                        mu=mu,
                        g2=g2,
                        sigma_n=sigma_n,
                        sigma_t=sigma_t,
                        sigma_w=sigma_w,
                        coarse_n=coarse_n,
                        fine_points_per_interval=fine_points_per_interval,
                    )
                    map_rebuilt = 1
                    accept_mstar += 1
                    mstar_accepted = 1

        # If map not rebuilt because m* rejected, keep current_map.
        # If you want, you could occasionally rebuild current_map every K steps,
        # but it is not necessary because x only changes within same m* block.

        # ---------------------------------------------
        # Block B: direct conditional resample x | current m*
        # ---------------------------------------------
        x_prop, st_prop, x_info = sample_x_from_fast_manifold_map(
            manifold_map=current_map,
            hitter=hitter,
            ev_star=current_ev_star,
            rng=rng,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )

        if st_prop is not None:
            current_x = float(x_prop)
            current_state = st_prop

        # ---------------------------------------------
        # Save
        # ---------------------------------------------
        if (it >= burn) and (((it - burn) % thin) == 0):
            meas_ll = log_p_mstar_given_mobs(
                current_raw_star,
                ev_obs_raw,
                sensor_sigmas=sensor_sigmas,
            )

            draws.append({
                "iter": int(it),

                "x_in": float(current_x),
                "psi_abs_rad": float(current_state["psi_abs"]),
                "psi_abs_deg": float(np.rad2deg(current_state["psi_abs"])),
                "psi_rel_rad": float(current_state["psi_rel"]),
                "psi_rel_deg": float(np.rad2deg(current_state["psi_rel"])),
                "e_x": float(current_state["e_x"]),
                "e_y": float(current_state["e_y"]),
                "omega_minus_rad_s": float(current_state["omega_minus"]),
                "omega_plus_rad_s": float(current_state["omega_plus"]),
                "D_in": float(current_state["D_in"]),
                "R_in": float(current_state["R_in"]),
                "r_x": float(current_state["r_x"]),
                "r_y": float(current_state["r_y"]),
                "Fn_res": float(current_state["Fn_res"]),
                "Ft_res": float(current_state["Ft_res"]),
                "Fw_res": float(current_state["Fw_res"]),
                "R_prof": float(current_state["R_prof"]),
                "log_prior_ex": float(current_state["log_prior_ex"]),
                "log_target_conditional": float(current_state["log_target"]),
                "log_p_mstar_given_mobs": float(meas_ll),
                "log_joint_proxy": float(current_state["log_target"] + meas_ll),

                "phi_rad": float(current_ev_star["phi"]),
                "phi_deg": float(np.rad2deg(current_ev_star["phi"])),
                "theta_abs_rad": float(current_ev_star["theta_abs"]),
                "theta_abs_deg": float(np.rad2deg(current_ev_star["theta_abs"])),
                "s_obs_fps": float(current_ev_star["s_obs"]),
                "vB_r_m_fps": float(current_ev_star["vB_r_m"]),
                "vB_z_m_fps": float(current_ev_star["vB_z_m"]),
                "vb_r_m_fps": float(current_ev_star["vb_r_m"]),
                "vb_z_m_fps": float(current_ev_star["vb_z_m"]),

                "mstar_accepted": int(mstar_accepted),
                "direct_x_resample_success": int(x_info["direct_x_resample_success"]),
                "map_rebuilt": int(map_rebuilt),
                "map_interval_count": int(x_info["map_interval_count"]),
                "map_total_width": float(x_info["map_total_width"]),
                "map_total_points": int(x_info["map_total_points"]),
            })

    posterior_df = pd.DataFrame(draws)

    meta = {
        "n_iter": int(n_iter),
        "burn": int(burn),
        "thin": int(thin),
        "n_saved": int(len(posterior_df)),
        "seed": int(seed),
        "init_try": int(init["init_try"]),
        "mstar_accept_rate": float(accept_mstar / max(n_iter, 1)),
        "mstar_valid_prop_rate": float(valid_mstar_prop / max(n_iter, 1)),
        "sigma_n": float(sigma_n),
        "sigma_t": float(sigma_t),
        "sigma_w": float(sigma_w),
        "coarse_n": int(coarse_n),
        "fine_points_per_interval": int(fine_points_per_interval),
    }

    summary_df = _summarize_draws(posterior_df)
    idata = _df_to_idata(posterior_df, meta=meta)

    return {
        "posterior_df": posterior_df,
        "summary_df": summary_df,
        "idata": idata,
        "meta": meta,
    }


# ---------------------------------------------------------
# Multi-event wrapper with progress monitoring
# ---------------------------------------------------------

def run_event_subset_fast_manifold(
    bip_model,
    hitters,
    event_indices,
    hitter_name_col="batter_name",
    seed=123,
    n_iter=2000,
    burn=500,
    thin=2,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=0.1,
    sigma_t=0.1,
    sigma_w=0.1,
    coarse_n=81,
    fine_points_per_interval=96,
    show_progress=True,
    skip_failures=True,
):
    """
    Run the fast manifold sampler on a subset of events.
    """
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    hitter_lookup = {h.name: h for h in hitters}
    results_by_event = {}
    summary_rows = []

    rng_master = np.random.default_rng(seed)
    iterator = event_indices
    if show_progress:
        iterator = tqdm(event_indices, desc="events", unit="event")

    for event_idx in iterator:
        row = bip_model.loc[event_idx]
        hitter_name = str(row[hitter_name_col]).lower()

        if hitter_name not in hitter_lookup:
            msg = f"No hitter object found for {hitter_name}"
            if skip_failures:
                print(f"Skipping event {event_idx}: {msg}")
                continue
            raise ValueError(msg)

        hitter = hitter_lookup[hitter_name]
        event_seed = int(rng_master.integers(1, 10_000_000))

        try:
            out = run_single_event_fast_manifold_chain(
                event_row=row,
                hitter=hitter,
                n_iter=n_iter,
                burn=burn,
                thin=thin,
                seed=event_seed,
                sensor_sigmas=sensor_sigmas,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
                coarse_n=coarse_n,
                fine_points_per_interval=fine_points_per_interval,
                show_progress=False,
            )

            results_by_event[event_idx] = out

            sdf = out["summary_df"].copy()
            row_summary = {"event_idx": int(event_idx), "batter_name": hitter.name}
            for _, r in sdf.iterrows():
                pname = r["parameter"]
                row_summary[f"{pname}_mean"] = r["mean"]
                row_summary[f"{pname}_sd"] = r["sd"]
                row_summary[f"{pname}_median"] = r["median"]
                row_summary[f"{pname}_q05"] = r["q05"]
                row_summary[f"{pname}_q95"] = r["q95"]

            row_summary["mstar_accept_rate"] = out["meta"]["mstar_accept_rate"]
            row_summary["mstar_valid_prop_rate"] = out["meta"]["mstar_valid_prop_rate"]
            summary_rows.append(row_summary)

        except Exception as e:
            if skip_failures:
                print(f"Skipping event {event_idx} due to error: {e}")
                continue
            raise

    summary_table = pd.DataFrame(summary_rows)

    return {
        "results_by_event": results_by_event,
        "summary_table": summary_table,
    }


# ---------------------------------------------------------
# Convenience run block for 10 events
# ---------------------------------------------------------

def run_fast_10_event_demo(
    bip_model,
    hitters,
    n_events=10,
    subset_seed=123,
    n_iter=2000,
    burn=500,
    thin=2,
):
    event_indices_10 = (
        bip_model
        .sample(n=n_events, random_state=subset_seed)
        .index
        .tolist()
    )

    print("Selected event indices:", event_indices_10)

    out = run_event_subset_fast_manifold(
        bip_model=bip_model,
        hitters=hitters,
        event_indices=event_indices_10,
        hitter_name_col="batter_name",
        seed=subset_seed,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        sensor_sigmas=default_sensor_sigmas(),
        psi_abs_min_deg=-85.0,
        psi_abs_max_deg=85.0,
        mu=0.5,
        g2=4.75,
        sigma_n=0.1,
        sigma_t=0.1,
        sigma_w=0.1,
        coarse_n=81,
        fine_points_per_interval=96,
        show_progress=True,
        skip_failures=True,
    )

    return out

# =========================================================
# DIAGNOSTICS-FIRST PATCH
#   - hard global cap: 0 <= e_x <= 0.6
#   - records why initialization fails
#   - compact posterior storage: x, psi_abs, psi_rel only
#   - 50-event summary runner
# =========================================================

import numpy as np
import pandas as pd
import arviz as az
from tqdm.auto import tqdm
from collections import Counter


# ---------------------------------------------------------
# Global physical cap for e_x
# ---------------------------------------------------------

EX_CAP_LO = 0.0
EX_CAP_HI = 0.6


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def _normal_logpdf_vec(x, mu, sd):
    z = (x - mu) / sd
    return -0.5 * z * z - np.log(sd) - 0.5 * np.log(2.0 * np.pi)


def _wrap_angle_pi_vec(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _safe_exp_normalized(logw):
    m = np.max(logw)
    if not np.isfinite(m):
        return np.zeros_like(logw)
    return np.exp(logw - m)


def _midpoint_cell_widths(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 1:
        return np.array([1.0], dtype=float)
    w = np.empty(n, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if n > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _series_top_reason(reason_counts):
    if not reason_counts:
        return "unknown"
    items = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


def _quantile_summary(vals, prefix):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q95": np.nan,
        }
    q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        f"{prefix}_median": float(q50),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q95": float(q95),
    }


# ---------------------------------------------------------
# Vectorized manifold eval WITH diagnostics and e_x cap
# Overrides prior version
# ---------------------------------------------------------

def vectorized_manifold_eval(
    x_arr,
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    denom_tol=DEFAULT_DENOM_TOL,
    ex_cap_lo=EX_CAP_LO,
    ex_cap_hi=EX_CAP_HI,
):
    """
    Exact 1-D manifold evaluation, vectorized over x, with explicit
    admissibility diagnostics and a global physical cap on e_x.
    """
    x_arr = np.asarray(x_arr, dtype=float)

    # hitter / constants
    L = float(hitter.bat_length)
    x_cm = float(hitter.x_cm)
    m_oz = float(hitter.m_ball_oz)
    M_oz = float(hitter.bat_weight_oz)
    I0 = float(hitter.I0_oz_in2)
    Iz = float(hitter.Iz_oz_in2)
    alpha = float(hitter.alpha)
    r_ball_in = float(hitter.r_ball_in)
    r_ball_ft = r_ball_in / 12.0

    # rebuilt event
    s_obs = float(ev_star["s_obs"])
    omega_minus = float(ev_star["omega_minus"])
    vB_r_m = float(ev_star["vB_r_m"])
    vB_z_m = float(ev_star["vB_z_m"])
    vb_r_m = float(ev_star["vb_r_m"])
    vb_z_m = float(ev_star["vb_z_m"])
    dv_r = float(ev_star["dv_r"])
    dv_z = float(ev_star["dv_z"])
    cos_theta = float(ev_star["cos_theta"])
    sin_theta = float(ev_star["sin_theta"])

    # bat profiles / recoil
    e_y = e_eff(x_arr, L=L, clip=True)
    R_in = Hitter.R(x_arr, L=L, clip=True)
    b = x_arr - x_cm

    r_y = m_oz * (1.0 / M_oz + (b ** 2) / I0)
    r_x = (m_oz * alpha / (1.0 + alpha)) * (
        1.0 / M_oz + (b ** 2) / I0 + (R_in ** 2) / Iz
    )

    coeff = (1.0 + e_y) / (1.0 + r_y)

    # analytic root psi_abs(x)
    A = vB_r_m + coeff * dv_r - s_obs * cos_theta
    B = vB_z_m + coeff * dv_z - s_obs * sin_theta
    amp = np.hypot(A, B)

    psi_abs = _wrap_angle_pi_vec(np.arctan2(-A, B))
    psi_lo = np.deg2rad(float(psi_abs_min_deg))
    psi_hi = np.deg2rad(float(psi_abs_max_deg))
    psi_on_support = (amp >= DEFAULT_ROOT_TOL) & (psi_abs >= psi_lo) & (psi_abs <= psi_hi)

    cp = np.cos(psi_abs)
    sp = np.sin(psi_abs)

    cos_theta_minus_psi = cos_theta * cp + sin_theta * sp
    sin_theta_minus_psi = sin_theta * cp - cos_theta * sp

    vB_n_m = vB_r_m * cp + vB_z_m * sp
    vB_t_m = -vB_r_m * sp + vB_z_m * cp

    vb_n_m = vb_r_m * cp + vb_z_m * sp
    vb_t_m = -vb_r_m * sp + vb_z_m * cp

    d_n = dv_r * cp + dv_z * sp
    d_t = -dv_r * sp + dv_z * cp

    psi_rel = np.arctan2(vb_t_m, vb_n_m)
    psi_regime_abs = np.abs(psi_rel)

    Vn_plus_obs = s_obs * cos_theta_minus_psi
    Vt_plus_obs = s_obs * sin_theta_minus_psi

    denom = alpha * (d_t - r_ball_ft * omega_minus)
    denom_ok = np.abs(denom) >= denom_tol

    e_x = np.full_like(x_arr, np.nan, dtype=float)
    numer_ex = ((Vt_plus_obs - vB_t_m) * (1.0 + r_x) * (1.0 + alpha))
    e_x[denom_ok] = numer_ex[denom_ok] / denom[denom_ok] - 1.0

    D_in = (R_in + r_ball_in) * np.sin(psi_rel)
    omega_plus = omega_minus + (
        ((vB_t_m - Vt_plus_obs) - (D_in / r_ball_in) * Vn_plus_obs)
        / (alpha * r_ball_ft)
    )

    # regime classification
    tan_psi_regime = np.tan(psi_regime_abs)
    regime_lower = mu * g2
    regime_upper = 3.5 * mu * (1.0 + e_y)

    regime_code = np.full_like(x_arr, 2, dtype=int)  # 0 stick-slip, 1 slip-stick-slip, 2 gross-slip
    regime_code[tan_psi_regime < regime_lower] = 0
    regime_code[(tan_psi_regime >= regime_lower) & (tan_psi_regime < regime_upper)] = 1

    regime_label = np.full_like(x_arr, "gross-slip", dtype=object)
    regime_label[regime_code == 0] = "stick-slip"
    regime_label[regime_code == 1] = "slip-stick-slip"

    # hard global e_x cap
    ex_global_ok = np.isfinite(e_x) & (e_x >= ex_cap_lo) & (e_x <= ex_cap_hi)

    # regime-dependent prior with hard global cap
    log_prior_ex = np.full_like(x_arr, -np.inf, dtype=float)

    m0 = (regime_code == 0) & ex_global_ok
    log_prior_ex[m0] = _normal_logpdf_vec(e_x[m0], 0.4, 0.1)

    m1 = (regime_code == 1) & ex_global_ok
    log_prior_ex[m1] = -np.log(ex_cap_hi - ex_cap_lo)

    # residuals
    Fn_res = vB_n_m + coeff * d_n - s_obs * cos_theta_minus_psi
    Ft_res = (
        vB_t_m
        + alpha * (1.0 + e_x) / ((1.0 + r_x) * (1.0 + alpha))
          * (d_t - r_ball_ft * omega_minus)
        - Vt_plus_obs
    )
    Fw_res = (
        alpha * r_ball_ft * (omega_plus - omega_minus)
        - (vB_t_m - Vt_plus_obs)
        + (D_in / r_ball_in) * Vn_plus_obs
    )

    R_prof = (
        (Fn_res / sigma_n) ** 2
        + (Ft_res / sigma_t) ** 2
        + (Fw_res / sigma_w) ** 2
    )

    log_target = log_prior_ex - 0.5 * R_prof

    vt_over_vn_abs = np.where(
        np.abs(Vn_plus_obs) >= denom_tol,
        np.abs(Vt_plus_obs) / np.abs(Vn_plus_obs),
        np.inf,
    )

    # admissibility masks
    psi_ok = psi_on_support
    denom_mask_ok = denom_ok
    vn_ok = Vn_plus_obs > 0.0
    vt_ok = vt_over_vn_abs < 1.0
    dn_ok = d_n > 0.0
    D_ok = np.isfinite(D_in) & (np.abs(D_in) <= (R_in + r_ball_in + 1e-8))
    regime_ok = regime_code != 2
    prior_ok = np.isfinite(log_prior_ex)
    numeric_ok = np.isfinite(log_target)

    admissible = (
        psi_ok
        & denom_mask_ok
        & vn_ok
        & vt_ok
        & dn_ok
        & D_ok
        & regime_ok
        & ex_global_ok
        & prior_ok
        & numeric_ok
    )

    # primary failure reason for diagnostics
    primary_reason = np.full(x_arr.shape, "admissible", dtype=object)
    unresolved = ~admissible

    mask = unresolved & (~psi_ok)
    primary_reason[mask] = "no_root_or_psi_support"
    unresolved &= psi_ok

    mask = unresolved & (~denom_mask_ok)
    primary_reason[mask] = "tangential_denom_singularity"
    unresolved &= denom_mask_ok

    mask = unresolved & (~vn_ok)
    primary_reason[mask] = "nonpositive_Vn_plus"
    unresolved &= vn_ok

    mask = unresolved & (~vt_ok)
    primary_reason[mask] = "vt_over_vn_ge_1"
    unresolved &= vt_ok

    mask = unresolved & (~dn_ok)
    primary_reason[mask] = "nonpositive_d_n"
    unresolved &= dn_ok

    mask = unresolved & (~D_ok)
    primary_reason[mask] = "invalid_D_geometry"
    unresolved &= D_ok

    mask = unresolved & (~regime_ok)
    primary_reason[mask] = "gross_slip_regime"
    unresolved &= regime_ok

    mask = unresolved & (~ex_global_ok)
    primary_reason[mask] = "e_x_out_of_bounds"
    unresolved &= ex_global_ok

    mask = unresolved & (~prior_ok)
    primary_reason[mask] = "invalid_e_x_prior"
    unresolved &= prior_ok

    mask = unresolved & (~numeric_ok)
    primary_reason[mask] = "nonfinite_log_target"

    return {
        "x": x_arr,
        "psi_abs": psi_abs,
        "psi_rel": psi_rel,
        "psi_regime_abs": psi_regime_abs,
        "e_x": e_x,
        "e_y": e_y,
        "omega_plus": omega_plus,
        "omega_minus": np.full_like(x_arr, omega_minus, dtype=float),
        "D_in": D_in,
        "R_in": R_in,
        "r_x": r_x,
        "r_y": r_y,
        "Vn_plus_obs": Vn_plus_obs,
        "Vt_plus_obs": Vt_plus_obs,
        "Fn_res": Fn_res,
        "Ft_res": Ft_res,
        "Fw_res": Fw_res,
        "R_prof": R_prof,
        "log_prior_ex": log_prior_ex,
        "log_target": log_target,
        "vt_over_vn_abs": vt_over_vn_abs,
        "tan_psi_regime": tan_psi_regime,
        "regime_lower": np.full_like(x_arr, regime_lower, dtype=float),
        "regime_upper": regime_upper,
        "regime_code": regime_code,
        "regime_label": regime_label,
        "admissible": admissible,
        "primary_reason": primary_reason,
    }


# ---------------------------------------------------------
# Build fast manifold map WITH coarse diagnostics
# Overrides prior version
# ---------------------------------------------------------

def build_fast_manifold_map(
    hitter,
    ev_star,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    coarse_n=81,
    fine_points_per_interval=96,
):
    x_lo = float(hitter.bat_length - 11.0)
    x_hi = float(hitter.bat_length)

    x_coarse = np.linspace(x_lo, x_hi, coarse_n)
    coarse = vectorized_manifold_eval(
        x_coarse,
        hitter=hitter,
        ev_star=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
    )

    coarse_reason_counts = dict(Counter(coarse["primary_reason"]))
    mask = coarse["admissible"]
    runs = _contiguous_true_runs(mask)

    if not runs:
        return {
            "intervals": [],
            "x": np.array([], dtype=float),
            "admissible_mask": np.array([], dtype=bool),
            "log_target": np.array([], dtype=float),
            "map": None,
            "total_width": 0.0,
            "total_points": 0,
            "coarse_reason_counts": coarse_reason_counts,
            "coarse_n_admissible": int(mask.sum()),
            "coarse_n_total": int(len(mask)),
        }

    refined_intervals = []
    for i0, i1 in runs:
        left_good = x_coarse[i0]
        right_good = x_coarse[i1]

        if i0 == 0:
            left_ref = float(left_good)
        else:
            left_bad = x_coarse[i0 - 1]
            left_ref = refine_interval_edge(
                x_bad=left_bad,
                x_good=left_good,
                hitter=hitter,
                ev_star=ev_star,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
            )

        if i1 == len(x_coarse) - 1:
            right_ref = float(right_good)
        else:
            lo = float(right_good)
            hi = float(x_coarse[i1 + 1])
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                st_mid = evaluate_x_state(
                    x_in=mid,
                    hitter=hitter,
                    ev_star=ev_star,
                    psi_abs_min_deg=psi_abs_min_deg,
                    psi_abs_max_deg=psi_abs_max_deg,
                    mu=mu,
                    g2=g2,
                    sigma_n=sigma_n,
                    sigma_t=sigma_t,
                    sigma_w=sigma_w,
                )
                if st_mid is None:
                    hi = mid
                else:
                    lo = mid
            right_ref = float(lo)

        if right_ref > left_ref:
            refined_intervals.append((left_ref, right_ref))

    if not refined_intervals:
        return {
            "intervals": [],
            "x": np.array([], dtype=float),
            "admissible_mask": np.array([], dtype=bool),
            "log_target": np.array([], dtype=float),
            "map": None,
            "total_width": 0.0,
            "total_points": 0,
            "coarse_reason_counts": coarse_reason_counts,
            "coarse_n_admissible": int(mask.sum()),
            "coarse_n_total": int(len(mask)),
        }

    x_chunks = []
    interval_id = []
    for k, (a, b) in enumerate(refined_intervals):
        n_k = max(16, fine_points_per_interval)
        xk = np.linspace(a, b, n_k)
        x_chunks.append(xk)
        interval_id.extend([k] * len(xk))

    x_fine = np.concatenate(x_chunks)
    interval_id = np.asarray(interval_id, dtype=int)

    fine = vectorized_manifold_eval(
        x_fine,
        hitter=hitter,
        ev_star=ev_star,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
    )

    total_width = float(sum(b - a for a, b in refined_intervals))

    return {
        "intervals": refined_intervals,
        "x": x_fine,
        "interval_id": interval_id,
        "admissible_mask": fine["admissible"],
        "log_target": fine["log_target"],
        "map": fine,
        "total_width": total_width,
        "total_points": int(len(x_fine)),
        "coarse_reason_counts": coarse_reason_counts,
        "coarse_n_admissible": int(mask.sum()),
        "coarse_n_total": int(len(mask)),
    }


# ---------------------------------------------------------
# Initialization with explicit failure recording
# ---------------------------------------------------------

def initialize_fast_chain_with_diagnostics(
    event_row,
    hitter,
    rng,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=DEFAULT_SIGMA_N,
    sigma_t=DEFAULT_SIGMA_T,
    sigma_w=DEFAULT_SIGMA_W,
    coarse_n=81,
    fine_points_per_interval=96,
    max_init_tries=300,
):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    ev_obs_raw = prepare_event_cache(event_row)

    init_reason_counter = Counter()
    init_attempt_log = []

    for init_try in range(1, max_init_tries + 1):
        raw_star = sample_measurement_star(ev_obs_raw, rng, sensor_sigmas=sensor_sigmas)
        ev_star = rebuild_observed_event(raw_star)

        core_vals = [
            ev_star["phi"], ev_star["theta"], ev_star["s_obs"],
            ev_star["omega_minus"],
            ev_star["vB_r_m"], ev_star["vB_z_m"],
            ev_star["vb_r_m"], ev_star["vb_z_m"],
        ]

        if not np.all(np.isfinite(core_vals)):
            init_reason_counter["nonfinite_rebuild"] += 1
            init_attempt_log.append({
                "init_try": init_try,
                "status": "failed",
                "top_reason": "nonfinite_rebuild",
                "reason_counts": {"nonfinite_rebuild": 1},
            })
            continue

        manifold_map = build_fast_manifold_map(
            hitter=hitter,
            ev_star=ev_star,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
            coarse_n=coarse_n,
            fine_points_per_interval=fine_points_per_interval,
        )

        if not manifold_map["intervals"]:
            rc = manifold_map["coarse_reason_counts"]
            for k, v in rc.items():
                if k != "admissible":
                    init_reason_counter[k] += v

            top_reason = _series_top_reason({k: v for k, v in rc.items() if k != "admissible"})
            if top_reason == "unknown":
                top_reason = "no_admissible_interval"

            init_attempt_log.append({
                "init_try": init_try,
                "status": "failed",
                "top_reason": top_reason,
                "reason_counts": rc,
                "coarse_n_admissible": manifold_map["coarse_n_admissible"],
                "coarse_n_total": manifold_map["coarse_n_total"],
            })
            continue

        x0, st0, xinfo = sample_x_from_fast_manifold_map(
            manifold_map=manifold_map,
            hitter=hitter,
            ev_star=ev_star,
            rng=rng,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )

        if st0 is not None:
            return {
                "success": True,
                "ev_obs_raw": ev_obs_raw,
                "raw_star": raw_star,
                "ev_star": ev_star,
                "x_in": float(x0),
                "state": st0,
                "manifold_map": manifold_map,
                "init_try": init_try,
                "init_reason_counter": dict(init_reason_counter),
                "init_attempt_log": init_attempt_log,
                "coarse_n_admissible": manifold_map["coarse_n_admissible"],
                "coarse_n_total": manifold_map["coarse_n_total"],
                "map_interval_count": len(manifold_map["intervals"]),
                "map_total_width": manifold_map["total_width"],
                "map_total_points": manifold_map["total_points"],
            }

        init_reason_counter["x_resample_failed"] += 1
        init_attempt_log.append({
            "init_try": init_try,
            "status": "failed",
            "top_reason": "x_resample_failed",
            "reason_counts": {"x_resample_failed": 1},
        })

    return {
        "success": False,
        "ev_obs_raw": ev_obs_raw,
        "init_try": max_init_tries,
        "init_reason_counter": dict(init_reason_counter),
        "init_attempt_log": init_attempt_log,
        "failure_top_reason": _series_top_reason(dict(init_reason_counter)),
    }


# ---------------------------------------------------------
# Compact single-event chain: keep only x / psi posteriors
# ---------------------------------------------------------

def run_single_event_fast_diagnostic_chain(
    event_row,
    hitter,
    n_iter=800,
    burn=200,
    thin=2,
    seed=123,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=0.1,
    sigma_t=0.1,
    sigma_w=0.1,
    coarse_n=61,
    fine_points_per_interval=80,
    show_progress=False,
):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    rng = np.random.default_rng(seed)

    init = initialize_fast_chain_with_diagnostics(
        event_row=event_row,
        hitter=hitter,
        rng=rng,
        sensor_sigmas=sensor_sigmas,
        psi_abs_min_deg=psi_abs_min_deg,
        psi_abs_max_deg=psi_abs_max_deg,
        mu=mu,
        g2=g2,
        sigma_n=sigma_n,
        sigma_t=sigma_t,
        sigma_w=sigma_w,
        coarse_n=coarse_n,
        fine_points_per_interval=fine_points_per_interval,
    )

    if not init["success"]:
        return {
            "success": False,
            "init_diagnostics": init,
        }

    ev_obs_raw = init["ev_obs_raw"]
    current_raw_star = init["raw_star"]
    current_ev_star = init["ev_star"]
    current_x = init["x_in"]
    current_state = init["state"]
    current_map = init["manifold_map"]

    draws = []
    accept_mstar = 0
    valid_mstar_prop = 0

    iterator = range(n_iter)
    if show_progress:
        iterator = tqdm(iterator, desc="event chain", unit="iter")

    for it in iterator:
        map_rebuilt = 0
        mstar_accepted = 0

        # m* update
        prop_raw_star = sample_measurement_star(ev_obs_raw, rng, sensor_sigmas=sensor_sigmas)
        prop_ev_star = rebuild_observed_event(prop_raw_star)

        prop_ok = np.all(np.isfinite([
            prop_ev_star["phi"], prop_ev_star["theta"], prop_ev_star["s_obs"],
            prop_ev_star["omega_minus"],
            prop_ev_star["vB_r_m"], prop_ev_star["vB_z_m"],
            prop_ev_star["vb_r_m"], prop_ev_star["vb_z_m"],
        ]))

        if prop_ok:
            prop_state_at_current_x = evaluate_x_state(
                x_in=current_x,
                hitter=hitter,
                ev_star=prop_ev_star,
                psi_abs_min_deg=psi_abs_min_deg,
                psi_abs_max_deg=psi_abs_max_deg,
                mu=mu,
                g2=g2,
                sigma_n=sigma_n,
                sigma_t=sigma_t,
                sigma_w=sigma_w,
            )

            if prop_state_at_current_x is not None:
                valid_mstar_prop += 1
                log_alpha = float(prop_state_at_current_x["log_target"] - current_state["log_target"])
                if np.log(rng.uniform()) < log_alpha:
                    current_raw_star = prop_raw_star
                    current_ev_star = prop_ev_star
                    current_state = prop_state_at_current_x

                    current_map = build_fast_manifold_map(
                        hitter=hitter,
                        ev_star=current_ev_star,
                        psi_abs_min_deg=psi_abs_min_deg,
                        psi_abs_max_deg=psi_abs_max_deg,
                        mu=mu,
                        g2=g2,
                        sigma_n=sigma_n,
                        sigma_t=sigma_t,
                        sigma_w=sigma_w,
                        coarse_n=coarse_n,
                        fine_points_per_interval=fine_points_per_interval,
                    )
                    map_rebuilt = 1
                    accept_mstar += 1
                    mstar_accepted = 1

        # x update from fast manifold map
        x_prop, st_prop, x_info = sample_x_from_fast_manifold_map(
            manifold_map=current_map,
            hitter=hitter,
            ev_star=current_ev_star,
            rng=rng,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
        )

        if st_prop is not None:
            current_x = float(x_prop)
            current_state = st_prop

        # save compact draws
        if (it >= burn) and (((it - burn) % thin) == 0):
            draws.append({
                "iter": int(it),
                "x_in": float(current_x),
                "psi_abs_deg": float(np.rad2deg(current_state["psi_abs"])),
                "psi_rel_deg": float(np.rad2deg(current_state["psi_rel"])),
                "mstar_accepted": int(mstar_accepted),
                "direct_x_resample_success": int(x_info["direct_x_resample_success"]),
                "map_rebuilt": int(map_rebuilt),
                "map_interval_count": int(x_info["map_interval_count"]),
                "map_total_width": float(x_info["map_total_width"]),
                "map_total_points": int(x_info["map_total_points"]),
            })

    posterior_df = pd.DataFrame(draws)

    summary_row = {
        **_quantile_summary(posterior_df["x_in"].values, "x_in"),
        **_quantile_summary(posterior_df["psi_abs_deg"].values, "psi_abs_deg"),
        **_quantile_summary(posterior_df["psi_rel_deg"].values, "psi_rel_deg"),
        "mstar_accept_rate": float(accept_mstar / max(n_iter, 1)),
        "mstar_valid_prop_rate": float(valid_mstar_prop / max(n_iter, 1)),
        "init_try": int(init["init_try"]),
        "coarse_n_admissible_init": int(init["coarse_n_admissible"]),
        "coarse_n_total_init": int(init["coarse_n_total"]),
        "map_interval_count_init": int(init["map_interval_count"]),
        "map_total_width_init": float(init["map_total_width"]),
        "map_total_points_init": int(init["map_total_points"]),
    }

    idata = az.from_dict(
        posterior={
            "x_in": posterior_df["x_in"].to_numpy(dtype=float)[None, :],
            "psi_abs_deg": posterior_df["psi_abs_deg"].to_numpy(dtype=float)[None, :],
            "psi_rel_deg": posterior_df["psi_rel_deg"].to_numpy(dtype=float)[None, :],
        }
    )

    return {
        "success": True,
        "posterior_df": posterior_df,
        "posterior_summary": pd.DataFrame([summary_row]),
        "idata": idata,
        "init_diagnostics": init,
    }


# ---------------------------------------------------------
# 50-event runner with success/failure tables
# ---------------------------------------------------------

def run_fast_diagnostics_subset(
    bip_model,
    hitters,
    event_indices,
    hitter_name_col="batter_name",
    seed=123,
    n_iter=800,
    burn=200,
    thin=2,
    sensor_sigmas=None,
    psi_abs_min_deg=-85.0,
    psi_abs_max_deg=85.0,
    mu=DEFAULT_MU,
    g2=DEFAULT_G2_BASEBALL,
    sigma_n=0.1,
    sigma_t=0.1,
    sigma_w=0.1,
    coarse_n=61,
    fine_points_per_interval=80,
    show_progress=True,
):
    if sensor_sigmas is None:
        sensor_sigmas = default_sensor_sigmas()

    hitter_lookup = {h.name: h for h in hitters}
    rng_master = np.random.default_rng(seed)

    success_rows = []
    failure_rows = []
    results_by_event = {}

    iterator = event_indices
    if show_progress:
        iterator = tqdm(event_indices, desc="50-event diagnostics", unit="event")

    for event_idx in iterator:
        row = bip_model.loc[event_idx]
        hitter_name = str(row[hitter_name_col]).lower()

        if hitter_name not in hitter_lookup:
            failure_rows.append({
                "event_idx": int(event_idx),
                "batter_name": hitter_name,
                "failure_top_reason": "missing_hitter_object",
            })
            continue

        hitter = hitter_lookup[hitter_name]
        event_seed = int(rng_master.integers(1, 10_000_000))

        out = run_single_event_fast_diagnostic_chain(
            event_row=row,
            hitter=hitter,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            seed=event_seed,
            sensor_sigmas=sensor_sigmas,
            psi_abs_min_deg=psi_abs_min_deg,
            psi_abs_max_deg=psi_abs_max_deg,
            mu=mu,
            g2=g2,
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            sigma_w=sigma_w,
            coarse_n=coarse_n,
            fine_points_per_interval=fine_points_per_interval,
            show_progress=False,
        )

        results_by_event[event_idx] = out

        if out["success"]:
            s = out["posterior_summary"].iloc[0].to_dict()
            init_diag = out["init_diagnostics"]

            success_rows.append({
                "event_idx": int(event_idx),
                "batter_name": hitter.name,
                **s,
                "init_failure_top_reason_before_success": _series_top_reason(init_diag["init_reason_counter"]),
                "init_reason_counter": str(init_diag["init_reason_counter"]),
            })
        else:
            init_diag = out["init_diagnostics"]
            failure_rows.append({
                "event_idx": int(event_idx),
                "batter_name": hitter.name,
                "failure_top_reason": init_diag["failure_top_reason"],
                "init_try": int(init_diag["init_try"]),
                "init_reason_counter": str(init_diag["init_reason_counter"]),
            })

    success_table = pd.DataFrame(success_rows)
    failure_table = pd.DataFrame(failure_rows)

    return {
        "results_by_event": results_by_event,
        "success_table": success_table,
        "failure_table": failure_table,
    }


if __name__ == "__main__":
    main()



