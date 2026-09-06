#!/usr/bin/env python3
"""Shared inference core for support/manifold samplers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


@dataclass
class CommonContext:
    production: Any
    bank: Any
    selected_events: pd.DataFrame
    calib_models: dict[str, Any]
    hitters_by_name: dict[str, Any]
    sensor_sigmas: dict[str, float]
    rng: np.random.Generator
    ey_spline: Any | None
    ey_spline_meta: dict[str, Any]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_arg_parser(default_root: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input-data-path", required=True)
    p.add_argument("--output-root", type=Path, default=default_root)
    p.add_argument("--max-events", type=int, default=200)
    p.add_argument("--event-ids", default="")
    p.add_argument("--random-seed", type=int, default=20260429)
    p.add_argument("--n-iter", type=int, default=700)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--target-successful-samples", type=int, default=500)
    p.add_argument("--max-iter-per-event", type=int, default=15000)
    p.add_argument("--proposal-sd-x", type=float, default=0.9)
    p.add_argument("--source-script-path", type=Path, required=False, default=None)
    p.add_argument("--production-script-path", type=Path, required=False, default=None)
    p.add_argument(
        "--ey-spline-draws-path",
        type=Path,
        default=Path(
            "/home/evangoforth03/Bayesian Research/MCMC 2/Outputs/refactor_exact_root_rhh/production/train_exports/posterior_draws_long_raw.parquet"
        ),
    )
    p.add_argument(
        "--hitter-meta-path",
        type=Path,
        default=Path("/home/evangoforth03/Bayesian Research/MCMC 2/hitter_meta.csv"),
    )
    p.add_argument("--compression-grid-size", type=int, default=250)
    p.add_argument("--comparison-support-draws-path", type=Path, default=None)
    p.add_argument("--comparison-manifold-draws-path", type=Path, default=None)
    return p


def build_context(args: argparse.Namespace) -> CommonContext:
    if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
        np.trapezoid = np.trapz  # type: ignore[attr-defined]

    # Compatibility shim for pickles that reference numpy._core.* paths.
    try:
        import numpy.core.numeric as _np_numeric  # type: ignore

        sys.modules.setdefault("numpy._core.numeric", _np_numeric)
    except Exception:
        pass

    bank = _load_module(Path(args.source_script_path), "bank_module_shared")
    production = _load_module(Path(args.production_script_path), "production_module_shared")

    base_df, bip_model, hitters_by_name = production.load_and_filter_events(Path(args.input_data_path), bank)
    _ = base_df  # retained for parity with production flow
    calib_models, _ = production.build_calibration_models(bip_model)
    event_ids = production.parse_event_ids(args.event_ids)
    selected = production.select_events(bip_model, event_ids, int(args.max_events), int(args.random_seed))
    sensor_sigmas = bank.default_sensor_sigmas()
    rng = np.random.default_rng(int(args.random_seed))
    return CommonContext(
        production=production,
        bank=bank,
        selected_events=selected,
        calib_models=calib_models,
        hitters_by_name=hitters_by_name,
        sensor_sigmas=sensor_sigmas,
        rng=rng,
        ey_spline=None,
        ey_spline_meta={},
    )


def _fit_ey_interpolator(draws_path: Path, hitter_meta_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Fit deterministic bivariate ey = f(x, L) from accepted production draws."""
    cols = ["batter_name", "x", "e_y_star", "accepted_draw"]
    draws = pd.read_parquet(draws_path, columns=cols)
    if "accepted_draw" in draws.columns:
        draws = draws[draws["accepted_draw"] == True]  # noqa: E712
    meta = pd.read_csv(hitter_meta_path)[["batter_name", "L_in"]].drop_duplicates("batter_name")
    df = draws.merge(meta, on="batter_name", how="left").dropna(subset=["x", "L_in", "e_y_star"])
    if df.empty:
        raise RuntimeError("No rows available to fit ey interpolator.")
    x = df["x"].to_numpy(dtype=float)
    L = df["L_in"].to_numpy(dtype=float)
    ey = df["e_y_star"].to_numpy(dtype=float)
    pts = np.column_stack([x, L])
    linear_interp = LinearNDInterpolator(pts, ey, fill_value=np.nan)
    nearest_interp = NearestNDInterpolator(pts, ey)
    meta_out = {"n_fit_rows": int(len(df)), "interpolator": "LinearND + NearestND fallback"}
    return linear_interp, nearest_interp, meta_out


def _deterministic_ey_from_interp(
    ey_interp: tuple[Any, Any], x_in: float, bat_length: float, ey_bounds: tuple[float, float]
) -> float:
    lin, nn = ey_interp
    ey_raw = lin(float(x_in), float(bat_length))
    if not np.isfinite(ey_raw):
        ey_raw = nn(float(x_in), float(bat_length))
    lo, hi = ey_bounds
    return float(np.clip(float(ey_raw), lo, hi))


def build_state_from_x(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    calib_model,
    x_in: float,
    sensor_sigmas: dict[str, float],
) -> dict[str, Any]:
    """Common preprocessing/target path: m* -> transforms -> exact-root state."""
    ev_star = ctx.bank.sample_measurement_star(ev_obs, rng=ctx.rng, sensor_sigmas=sensor_sigmas)
    logp_m = float(ctx.bank.log_p_mstar_given_mobs(ev_star, ev_obs, sensor_sigmas=sensor_sigmas))
    trans = ctx.production.build_transformed_measurements(ev_star, calib_model, ctx.bank)
    e_y_star, logp_ey = ctx.production.sample_e_y_star(float(x_in), hitter, ctx.bank, ctx.rng)
    st = ctx.production.solve_exact_root_state(
        ev_obs=ev_obs,
        ev_star=ev_star,
        trans=trans,
        hitter=hitter,
        x_in=float(x_in),
        e_y_star=float(e_y_star),
        logp_measure=float(logp_m),
        logp_ey=float(logp_ey),
        bank_module=ctx.bank,
    )
    st.update(
        {
            "event_id": int(ev_obs["event_id"]),
            "batter_name": str(ev_obs["batter_name"]),
            "phi_star": trans.get("phi_star_deg", np.nan),
            "delta_star": trans.get("delta_star_deg", np.nan),
            "d_tilde": trans.get("attack_direction_calib_deg", np.nan),
            "a_tilde": trans.get("attack_angle_calib_to_spray_deg", np.nan),
            "v_ss_tilde": trans.get("bat_speed_calib_to_spray_mph", np.nan),
            "log_q_proposal": float(-math.log(11.0) + logp_m + logp_ey),
        }
    )
    return st


def build_state_from_fixed_measurement(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    x_in: float,
    logp_measure: float,
    deterministic_ey: bool = False,
) -> dict[str, Any]:
    """State builder when one m* draw is fixed and x is varied."""
    if deterministic_ey:
        if ctx.ey_spline is None:
            raise RuntimeError("deterministic_ey=True but ey spline is missing")
        L = float(hitter.bat_length)
        lin, nn = ctx.ey_spline
        ey_raw = lin(float(x_in), L)
        if not np.isfinite(ey_raw):
            ey_raw = nn(float(x_in), L)
        ey_raw = float(ey_raw)
        lo, hi = ctx.production.EY_BOUNDS
        e_y_star = float(np.clip(ey_raw, lo, hi))
        logp_ey = 0.0
    else:
        e_y_star, logp_ey = ctx.production.sample_e_y_star(float(x_in), hitter, ctx.bank, ctx.rng)
    st = ctx.production.solve_exact_root_state(
        ev_obs=ev_obs,
        ev_star={},
        trans=trans,
        hitter=hitter,
        x_in=float(x_in),
        e_y_star=float(e_y_star),
        logp_measure=float(logp_measure),
        logp_ey=float(logp_ey),
        bank_module=ctx.bank,
    )
    st.update(
        {
            "event_id": int(ev_obs["event_id"]),
            "batter_name": str(ev_obs["batter_name"]),
            "phi_star": trans.get("phi_star_deg", np.nan),
            "delta_star": trans.get("delta_star_deg", np.nan),
            "d_tilde": trans.get("attack_direction_calib_deg", np.nan),
            "a_tilde": trans.get("attack_angle_calib_to_spray_deg", np.nan),
            "v_ss_tilde": trans.get("bat_speed_calib_to_spray_mph", np.nan),
            "log_q_proposal": float(-math.log(11.0) + logp_measure + logp_ey),
        }
    )
    return st


def run_support_sampler(ctx: CommonContext, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Broad-support proposal over x with admissibility-gated MH."""
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for _, row in ctx.selected_events.iterrows():
        t0 = time.perf_counter()
        bname = str(row["batter_name"])
        hitter = ctx.hitters_by_name.get(bname)
        calib_model = ctx.calib_models.get(bname)
        if hitter is None or calib_model is None:
            continue
        ev_obs = ctx.production.event_to_observed_dict(row, ctx.bank)

        lo, hi = hitter.bat_length - 11.0, hitter.bat_length
        curr = None
        for _ in range(3000):
            x0 = float(ctx.rng.uniform(lo, hi))
            st0 = build_state_from_x(ctx, ev_obs, hitter, calib_model, x0, ctx.sensor_sigmas)
            if st0.get("accepted_physics", 0) == 1 and np.isfinite(st0.get("log_target", np.nan)):
                curr = st0
                break
        if curr is None:
            timing_rows.append(
                {
                    "event_id": int(ev_obs["event_id"]),
                    "batter_name": str(ev_obs["batter_name"]),
                    "sampler": "support",
                    "target_successful_samples": int(args.target_successful_samples),
                    "successful_samples_obtained": 0,
                    "iterations_executed": 0,
                    "runtime_seconds": float(time.perf_counter() - t0),
                    "status": "failed_initialization",
                }
            )
            continue

        target_n = int(args.target_successful_samples)
        max_iter = int(args.max_iter_per_event)
        kept_n = 0
        it = 0
        while kept_n < target_n and it < max_iter:
            x_prop = ctx.production._reflect_to_bounds(
                float(curr["x"]) + float(ctx.rng.normal(0.0, float(args.proposal_sd_x))), lo, hi
            )
            prop = build_state_from_x(ctx, ev_obs, hitter, calib_model, x_prop, ctx.sensor_sigmas)
            moved = 0
            if prop.get("accepted_physics", 0) == 1 and np.isfinite(prop.get("log_target", np.nan)):
                log_alpha = (prop["log_target"] - curr["log_target"]) + (
                    curr.get("log_q_proposal", 0.0) - prop.get("log_q_proposal", 0.0)
                )
                if np.log(ctx.rng.uniform()) < min(0.0, log_alpha):
                    curr = prop
                    moved = 1
            if curr.get("accepted_physics", 0) == 1 and np.isfinite(curr.get("log_target", np.nan)):
                kept = dict(curr)
                kept["iter"] = int(it)
                kept["accepted_move"] = int(moved)
                kept["sampler"] = "support"
                rows.append(kept)
                kept_n += 1
            it += 1
        timing_rows.append(
            {
                "event_id": int(ev_obs["event_id"]),
                "batter_name": str(ev_obs["batter_name"]),
                "sampler": "support",
                "target_successful_samples": target_n,
                "successful_samples_obtained": int(kept_n),
                "iterations_executed": int(it),
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "ok" if kept_n >= target_n else "max_iter_reached",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(timing_rows)


def _admissible_weighted_x(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    calib_model,
    n_grid: int = 10,
) -> tuple[float | None, dict[str, Any] | None]:
    _ = n_grid
    return None, None


def _admissible_at_x_for_interval(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    x_in: float,
) -> bool:
    e_y_star, logp_ey = ctx.production.sample_e_y_star(float(x_in), hitter, ctx.bank, ctx.rng)
    st = ctx.production.solve_exact_root_state(
        ev_obs=ev_obs,
        ev_star={},
        trans=trans,
        hitter=hitter,
        x_in=float(x_in),
        e_y_star=float(e_y_star),
        logp_measure=0.0,
        logp_ey=float(logp_ey),
        bank_module=ctx.bank,
    )
    return bool(st.get("accepted_physics", 0) == 1 and np.isfinite(st.get("log_target", np.nan)))


def _bisect_boundary(
    pred,
    x_good: float,
    x_bad: float,
    n_iter: int = 30,
) -> float:
    lo = float(x_good)
    hi = float(x_bad)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def _trace_interval_from_seed(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    seed_x: float,
    lo: float,
    hi: float,
    step: float = 0.05,
) -> tuple[float, float]:
    pred = lambda xx: _admissible_at_x_for_interval(ctx, ev_obs, hitter, trans, float(xx))
    x0 = float(seed_x)

    # Left boundary
    x_good = x0
    x_try = x0
    while True:
        nxt = max(lo, x_try - step)
        if nxt == x_try:
            left = lo if pred(lo) else x_good
            break
        if pred(nxt):
            x_good = nxt
            x_try = nxt
            continue
        left = _bisect_boundary(pred, x_good, nxt)
        break

    # Right boundary
    x_good = x0
    x_try = x0
    while True:
        nxt = min(hi, x_try + step)
        if nxt == x_try:
            right = hi if pred(hi) else x_good
            break
        if pred(nxt):
            x_good = nxt
            x_try = nxt
            continue
        right = _bisect_boundary(pred, x_good, nxt)
        break

    return float(left), float(right)


def _discover_intervals_no_grid(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    seed_x: float,
) -> list[tuple[float, float]]:
    lo, hi = hitter.bat_length - 11.0, hitter.bat_length
    step = 0.05
    first = _trace_interval_from_seed(ctx, ev_obs, hitter, trans, float(seed_x), lo, hi)
    intervals = [first]
    pred = lambda xx: _admissible_at_x_for_interval(ctx, ev_obs, hitter, trans, float(xx))

    # Search remaining support in 0.05 increments for additional admissible components.
    a1, b1 = first
    x = lo
    while x <= hi:
        if (a1 - 1e-3) <= x <= (b1 + 1e-3):
            x += step
            continue
        if pred(x):
            second = _trace_interval_from_seed(ctx, ev_obs, hitter, trans, x, lo, hi, step=step)
            c, d = second
            # Keep only disconnected interval(s) distinct from first.
            if d < a1 - 1e-3 or c > b1 + 1e-3:
                intervals.append(second)
            break
        x += step

    intervals = sorted(intervals, key=lambda t: t[0])
    # Keep at most two intervals per expected geometry.
    return intervals[:2]


def _sample_x_from_intervals(ctx: CommonContext, intervals: list[tuple[float, float]]) -> float:
    widths = np.array([max(0.0, b - a) for a, b in intervals], dtype=float)
    if widths.sum() <= 0:
        a, b = intervals[0]
        return float(0.5 * (a + b))
    p = widths / widths.sum()
    idx = int(ctx.rng.choice(np.arange(len(intervals)), p=p))
    a, b = intervals[idx]
    return float(ctx.rng.uniform(a, b))


def _find_measurement_with_intervals(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    calib_model,
    max_mstar_tries: int = 500,
    max_seed_tries: int = 300,
) -> tuple[dict[str, Any] | None, float | None, list[tuple[float, float]]]:
    lo, hi = hitter.bat_length - 11.0, hitter.bat_length
    for _ in range(max_mstar_tries):
        ev_star = ctx.bank.sample_measurement_star(ev_obs, rng=ctx.rng, sensor_sigmas=ctx.sensor_sigmas)
        logp_m = float(ctx.bank.log_p_mstar_given_mobs(ev_star, ev_obs, sensor_sigmas=ctx.sensor_sigmas))
        trans = ctx.production.build_transformed_measurements(ev_star, calib_model, ctx.bank)

        seed_x = None
        for _ in range(max_seed_tries):
            x = float(ctx.rng.uniform(lo, hi))
            if _admissible_at_x_for_interval(ctx, ev_obs, hitter, trans, x):
                seed_x = x
                break
        if seed_x is None:
            continue
        intervals = _discover_intervals_no_grid(ctx, ev_obs, hitter, trans, seed_x)
        intervals = [(a, b) for a, b in intervals if (b - a) > 1e-4]
        if intervals:
            return trans, logp_m, intervals
    return None, None, []


def run_manifold_sampler(ctx: CommonContext, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample m* until admissibility is found, freeze m*, trace admissible x-intervals
    directly (no global grid search), then sample only from discovered intervals.
    """
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for _, row in ctx.selected_events.iterrows():
        t0 = time.perf_counter()
        bname = str(row["batter_name"])
        hitter = ctx.hitters_by_name.get(bname)
        calib_model = ctx.calib_models.get(bname)
        if hitter is None or calib_model is None:
            continue
        ev_obs = ctx.production.event_to_observed_dict(row, ctx.bank)
        trans, logp_m, intervals = _find_measurement_with_intervals(ctx, ev_obs, hitter, calib_model)
        if trans is None or logp_m is None or not intervals:
            timing_rows.append(
                {
                    "event_id": int(ev_obs["event_id"]),
                    "batter_name": str(ev_obs["batter_name"]),
                    "sampler": "manifold",
                    "target_successful_samples": int(args.target_successful_samples),
                    "successful_samples_obtained": 0,
                    "iterations_executed": 0,
                    "runtime_seconds": float(time.perf_counter() - t0),
                    "status": "failed_mstar_interval_discovery",
                    "n_intervals": 0,
                    "intervals_json": "[]",
                }
            )
            continue

        target_n = int(args.target_successful_samples)
        max_iter = int(args.max_iter_per_event)
        kept_n = 0
        it = 0
        curr = None
        for _ in range(5000):
            x0 = _sample_x_from_intervals(ctx, intervals)
            st0 = build_state_from_fixed_measurement(
                ctx=ctx,
                ev_obs=ev_obs,
                hitter=hitter,
                trans=trans,
                x_in=float(x0),
                logp_measure=float(logp_m),
                deterministic_ey=False,
            )
            if st0.get("accepted_physics", 0) == 1 and np.isfinite(st0.get("log_target", np.nan)):
                curr = st0
                curr["x"] = float(x0)
                break
        if curr is None:
            timing_rows.append(
                {
                    "event_id": int(ev_obs["event_id"]),
                    "batter_name": str(ev_obs["batter_name"]),
                    "sampler": "manifold",
                    "target_successful_samples": target_n,
                    "successful_samples_obtained": 0,
                    "iterations_executed": 0,
                    "runtime_seconds": float(time.perf_counter() - t0),
                    "status": "failed_initialization_from_intervals",
                    "n_intervals": int(len(intervals)),
                    "intervals_json": json.dumps([[float(a), float(b)] for a, b in intervals]),
                }
            )
            continue

        while kept_n < target_n and it < max_iter:
            x_draw = _sample_x_from_intervals(ctx, intervals)
            st = build_state_from_fixed_measurement(
                ctx=ctx,
                ev_obs=ev_obs,
                hitter=hitter,
                trans=trans,
                x_in=float(x_draw),
                logp_measure=float(logp_m),
                deterministic_ey=False,
            )
            moved = 0
            if st.get("accepted_physics", 0) != 1 or not np.isfinite(st.get("log_target", np.nan)):
                it += 1
                continue
            log_alpha = (st["log_target"] - curr["log_target"]) + (
                curr.get("log_q_proposal", 0.0) - st.get("log_q_proposal", 0.0)
            )
            if np.log(ctx.rng.uniform()) < min(0.0, log_alpha):
                curr = st
                curr["x"] = float(x_draw)
                moved = 1
            kept = dict(curr)
            kept["iter"] = int(it)
            kept["x"] = float(curr.get("x", x_draw))
            kept["accepted_move"] = int(moved)
            kept["sampler"] = "manifold"
            rows.append(kept)
            kept_n += 1
            it += 1
        timing_rows.append(
            {
                "event_id": int(ev_obs["event_id"]),
                "batter_name": str(ev_obs["batter_name"]),
                "sampler": "manifold",
                "target_successful_samples": target_n,
                "successful_samples_obtained": int(kept_n),
                "iterations_executed": int(it),
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "ok" if kept_n >= target_n else "max_iter_reached",
                "n_intervals": int(len(intervals)),
                "intervals_json": json.dumps([[float(a), float(b)] for a, b in intervals]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(timing_rows)


def _build_state_from_fixed_measurement_det_ey(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    x_in: float,
    logp_measure: float,
    ey_interp: tuple[Any, Any],
) -> dict[str, Any]:
    e_y_star = _deterministic_ey_from_interp(
        ey_interp, float(x_in), float(hitter.bat_length), tuple(ctx.production.EY_BOUNDS)
    )
    logp_ey = 0.0
    st = ctx.production.solve_exact_root_state(
        ev_obs=ev_obs,
        ev_star={},
        trans=trans,
        hitter=hitter,
        x_in=float(x_in),
        e_y_star=float(e_y_star),
        logp_measure=float(logp_measure),
        logp_ey=float(logp_ey),
        bank_module=ctx.bank,
    )
    st.update(
        {
            "event_id": int(ev_obs["event_id"]),
            "batter_name": str(ev_obs["batter_name"]),
            "phi_star": trans.get("phi_star_deg", np.nan),
            "delta_star": trans.get("delta_star_deg", np.nan),
            "d_tilde": trans.get("attack_direction_calib_deg", np.nan),
            "a_tilde": trans.get("attack_angle_calib_to_spray_deg", np.nan),
            "v_ss_tilde": trans.get("bat_speed_calib_to_spray_mph", np.nan),
            "log_q_proposal": float(-math.log(11.0) + logp_measure + logp_ey),
            "e_y_det": float(e_y_star),
        }
    )
    return st


def _admissible_at_x_for_interval_det_ey(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    x_in: float,
    ey_interp: tuple[Any, Any],
) -> bool:
    st = _build_state_from_fixed_measurement_det_ey(
        ctx=ctx,
        ev_obs=ev_obs,
        hitter=hitter,
        trans=trans,
        x_in=float(x_in),
        logp_measure=0.0,
        ey_interp=ey_interp,
    )
    return bool(st.get("accepted_physics", 0) == 1 and np.isfinite(st.get("log_target", np.nan)))


def _trace_interval_from_seed_det_ey(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    seed_x: float,
    lo: float,
    hi: float,
    ey_interp: tuple[Any, Any],
    step: float = 0.05,
) -> tuple[float, float]:
    pred = lambda xx: _admissible_at_x_for_interval_det_ey(ctx, ev_obs, hitter, trans, float(xx), ey_interp)
    x0 = float(seed_x)
    x_good = x0
    x_try = x0
    while True:
        nxt = max(lo, x_try - step)
        if nxt == x_try:
            left = lo if pred(lo) else x_good
            break
        if pred(nxt):
            x_good = nxt
            x_try = nxt
            continue
        left = _bisect_boundary(pred, x_good, nxt)
        break
    x_good = x0
    x_try = x0
    while True:
        nxt = min(hi, x_try + step)
        if nxt == x_try:
            right = hi if pred(hi) else x_good
            break
        if pred(nxt):
            x_good = nxt
            x_try = nxt
            continue
        right = _bisect_boundary(pred, x_good, nxt)
        break
    return float(left), float(right)


def _discover_intervals_no_grid_det_ey(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    trans: dict[str, Any],
    seed_x: float,
    ey_interp: tuple[Any, Any],
) -> list[tuple[float, float]]:
    lo, hi = hitter.bat_length - 11.0, hitter.bat_length
    step = 0.05
    first = _trace_interval_from_seed_det_ey(ctx, ev_obs, hitter, trans, float(seed_x), lo, hi, ey_interp, step=step)
    intervals = [first]
    pred = lambda xx: _admissible_at_x_for_interval_det_ey(ctx, ev_obs, hitter, trans, float(xx), ey_interp)
    a1, b1 = first
    x = lo
    while x <= hi:
        if (a1 - 1e-3) <= x <= (b1 + 1e-3):
            x += step
            continue
        if pred(x):
            second = _trace_interval_from_seed_det_ey(ctx, ev_obs, hitter, trans, x, lo, hi, ey_interp, step=step)
            c, d = second
            if d < a1 - 1e-3 or c > b1 + 1e-3:
                intervals.append(second)
            break
        x += step
    return sorted(intervals, key=lambda t: t[0])[:2]


def _find_measurement_with_intervals_det_ey(
    ctx: CommonContext,
    ev_obs: dict[str, Any],
    hitter,
    calib_model,
    ey_interp: tuple[Any, Any],
    max_mstar_tries: int = 150,
    max_seed_tries: int = 120,
) -> tuple[dict[str, Any] | None, float | None, list[tuple[float, float]]]:
    lo, hi = hitter.bat_length - 11.0, hitter.bat_length
    for _ in range(max_mstar_tries):
        ev_star = ctx.bank.sample_measurement_star(ev_obs, rng=ctx.rng, sensor_sigmas=ctx.sensor_sigmas)
        logp_m = float(ctx.bank.log_p_mstar_given_mobs(ev_star, ev_obs, sensor_sigmas=ctx.sensor_sigmas))
        trans = ctx.production.build_transformed_measurements(ev_star, calib_model, ctx.bank)
        seed_x = None
        for _ in range(max_seed_tries):
            x = float(ctx.rng.uniform(lo, hi))
            if _admissible_at_x_for_interval_det_ey(ctx, ev_obs, hitter, trans, x, ey_interp):
                seed_x = x
                break
        if seed_x is None:
            continue
        intervals = _discover_intervals_no_grid_det_ey(ctx, ev_obs, hitter, trans, seed_x, ey_interp)
        intervals = [(a, b) for a, b in intervals if (b - a) > 1e-4]
        if intervals:
            return trans, logp_m, intervals
    return None, None, []


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    target = float(q) * cw[-1]
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _event_comparison_from_density(
    density_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    if density_df.empty or sampled_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ev, g in density_df.groupby("event_id"):
        samp = sampled_df[sampled_df["event_id"] == ev]
        if samp.empty:
            continue
        w = g["pi_can_norm"].to_numpy(dtype=float)
        if not np.isfinite(w).all() or np.nansum(w) <= 0:
            continue
        w = w / np.nansum(w)
        row = {"event_id": int(ev), "comparison_method": method_name}
        for col in ["x", "psi", "e_x", "e_y_star"]:
            if col not in g.columns or col not in samp.columns:
                continue
            gv = g[col].to_numpy(dtype=float)
            sv = samp[col].to_numpy(dtype=float)
            row[f"{col}_compression_mean"] = float(np.nansum(gv * w))
            row[f"{col}_compression_q50"] = _weighted_quantile(gv, w, 0.5)
            row[f"{col}_sample_mean"] = float(np.nanmean(sv))
            row[f"{col}_sample_q50"] = float(np.nanmedian(sv))
        rows.append(row)
    return pd.DataFrame(rows)


def run_manifold_compression_sampler(
    ctx: CommonContext, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Physics-informed admissible manifold compression:
    1) find admissible m*, 2) discover admissible x intervals, 3) tabulate g(x),
    4) compute canonical density pi_can(x) ∝ pi0(gamma(x)) / |det A(gamma(x))|.
    """
    ey_interp = _fit_ey_interpolator(Path(args.ey_spline_draws_path), Path(args.hitter_meta_path))[:2]
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, row in ctx.selected_events.iterrows():
        t0 = time.perf_counter()
        bname = str(row["batter_name"])
        hitter = ctx.hitters_by_name.get(bname)
        calib_model = ctx.calib_models.get(bname)
        if hitter is None or calib_model is None:
            continue
        ev_obs = ctx.production.event_to_observed_dict(row, ctx.bank)
        trans, logp_m, intervals = _find_measurement_with_intervals_det_ey(
            ctx, ev_obs, hitter, calib_model, ey_interp=ey_interp
        )
        if trans is None or logp_m is None or not intervals:
            timing_rows.append(
                {
                    "event_id": int(ev_obs["event_id"]),
                    "batter_name": str(ev_obs["batter_name"]),
                    "sampler": "manifold_compression",
                    "runtime_seconds": float(time.perf_counter() - t0),
                    "status": "failed_mstar_interval_discovery",
                    "n_intervals": 0,
                    "intervals_json": "[]",
                }
            )
            continue
        grid_n = max(50, int(args.compression_grid_size))
        event_rows_start = len(rows)
        for branch_id, (a, b) in enumerate(intervals):
            xs = np.linspace(float(a), float(b), grid_n)
            for x_in in xs:
                st = _build_state_from_fixed_measurement_det_ey(
                    ctx=ctx,
                    ev_obs=ev_obs,
                    hitter=hitter,
                    trans=trans,
                    x_in=float(x_in),
                    logp_measure=float(logp_m),
                    ey_interp=ey_interp,
                )
                if st.get("accepted_physics", 0) != 1 or not np.isfinite(st.get("log_target", np.nan)):
                    continue
                psi_rad = float(st.get("psi_rad", np.nan))
                A = float(st.get("A", np.nan))
                B = float(st.get("B", np.nan))
                dpsi_fn = abs((-A * math.sin(psi_rad)) + (B * math.cos(psi_rad)))
                r_x = float(st.get("r_x_x", np.nan))
                denom_ft = abs(float(st.get("Ft_denom", np.nan)))
                dft_dex = denom_ft / max(1e-9, (1.0 + r_x) * (1.0 + float(hitter.alpha)))
                dfw_domega = abs(float(st.get("Fw_denom", np.nan)))
                detA_abs = dpsi_fn * dft_dex * dfw_domega
                if not np.isfinite(detA_abs) or detA_abs <= 1e-12:
                    continue
                pi0 = float(np.exp(st["log_target"]))
                pi_can_unnorm = pi0 / detA_abs
                rows.append(
                    {
                        "event_id": int(ev_obs["event_id"]),
                        "batter_name": str(ev_obs["batter_name"]),
                        "branch_id": int(branch_id),
                        "x": float(x_in),
                        "psi": float(st.get("psi", np.nan)),
                        "e_x": float(st.get("e_x", np.nan)),
                        "omega_plus": float(st.get("omega_plus", np.nan)),
                        "e_y_star": float(st.get("e_y_det", st.get("e_y_star", np.nan))),
                        "detA_abs": float(detA_abs),
                        "pi0": float(pi0),
                        "pi_can_unnorm": float(pi_can_unnorm),
                    }
                )
        event_rows = rows[event_rows_start:]
        if not event_rows:
            timing_rows.append(
                {
                    "event_id": int(ev_obs["event_id"]),
                    "batter_name": str(ev_obs["batter_name"]),
                    "sampler": "manifold_compression",
                    "runtime_seconds": float(time.perf_counter() - t0),
                    "status": "no_admissible_grid_points",
                    "n_intervals": int(len(intervals)),
                    "intervals_json": json.dumps([[float(a), float(b)] for a, b in intervals]),
                }
            )
            continue
        weights = np.array([r["pi_can_unnorm"] for r in event_rows], dtype=float)
        z = float(np.trapz(weights, dx=1.0)) if len(weights) > 1 else float(weights[0])
        if not np.isfinite(z) or z <= 0:
            z = float(np.sum(weights))
        z = max(z, 1e-15)
        for r in event_rows:
            r["pi_can_norm"] = float(r["pi_can_unnorm"] / z)
        mass_by_branch: dict[int, float] = {}
        for r in event_rows:
            bid = int(r["branch_id"])
            mass_by_branch[bid] = mass_by_branch.get(bid, 0.0) + float(r["pi_can_norm"])
        summary_rows.append(
            {
                "event_id": int(ev_obs["event_id"]),
                "batter_name": str(ev_obs["batter_name"]),
                "n_intervals": int(len(intervals)),
                "intervals_json": json.dumps([[float(a), float(b)] for a, b in intervals]),
                "n_grid_points_kept": int(len(event_rows)),
                "mass_by_branch_json": json.dumps({str(k): float(v) for k, v in mass_by_branch.items()}),
                "status": "ok",
            }
        )
        timing_rows.append(
            {
                "event_id": int(ev_obs["event_id"]),
                "batter_name": str(ev_obs["batter_name"]),
                "sampler": "manifold_compression",
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "ok",
                "n_intervals": int(len(intervals)),
                "intervals_json": json.dumps([[float(a), float(b)] for a, b in intervals]),
            }
        )
    density_df = pd.DataFrame(rows)
    timing_df = pd.DataFrame(timing_rows)
    summary_df = pd.DataFrame(summary_rows)
    comparison_frames: list[pd.DataFrame] = []
    if args.comparison_support_draws_path and Path(args.comparison_support_draws_path).exists():
        support_df = pd.read_parquet(Path(args.comparison_support_draws_path))
        comparison_frames.append(_event_comparison_from_density(density_df, support_df, "support"))
    if args.comparison_manifold_draws_path and Path(args.comparison_manifold_draws_path).exists():
        manifold_df = pd.read_parquet(Path(args.comparison_manifold_draws_path))
        comparison_frames.append(_event_comparison_from_density(density_df, manifold_df, "manifold"))
    comparison_df = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    return density_df, timing_df, summary_df, comparison_df


def write_compression_outputs(
    density_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_root: Path,
    name: str,
    args: argparse.Namespace,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    density_df.to_parquet(out_root / f"{name}_branch_density.parquet", index=False)
    density_df.to_csv(out_root / f"{name}_branch_density.csv", index=False)
    timing_df.to_parquet(out_root / f"{name}_event_timing.parquet", index=False)
    timing_df.to_csv(out_root / f"{name}_event_timing.csv", index=False)
    summary_df.to_parquet(out_root / f"{name}_event_summary.parquet", index=False)
    summary_df.to_csv(out_root / f"{name}_event_summary.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_parquet(out_root / f"{name}_comparison.parquet", index=False)
        comparison_df.to_csv(out_root / f"{name}_comparison.csv", index=False)
    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    manifest = {
        "n_density_rows": int(len(density_df)),
        "n_events_with_density": int(density_df["event_id"].nunique()) if "event_id" in density_df.columns else 0,
        "timed_events": int(len(timing_df)),
        "mean_runtime_seconds": float(timing_df["runtime_seconds"].mean()) if len(timing_df) else None,
        "args": args_json,
    }
    (out_root / f"{name}_run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def write_outputs(df: pd.DataFrame, timing_df: pd.DataFrame, out_root: Path, name: str, args: argparse.Namespace) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_root / f"{name}_posterior_draws.parquet", index=False)
    timing_df.to_parquet(out_root / f"{name}_event_timing.parquet", index=False)
    timing_df.to_csv(out_root / f"{name}_event_timing.csv", index=False)
    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "n_rows": int(len(df)),
        "n_events": int(df["event_id"].nunique()) if "event_id" in df.columns and len(df) else 0,
        "timed_events": int(len(timing_df)),
        "mean_runtime_seconds": float(timing_df["runtime_seconds"].mean()) if len(timing_df) else None,
        "args": args_json,
    }
    (out_root / f"{name}_run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
