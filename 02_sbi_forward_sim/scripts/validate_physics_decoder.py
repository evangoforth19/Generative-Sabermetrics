#!/usr/bin/env python3
"""Sanity checks for the production-aligned physics decoder (no retraining)."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

_MMC2_ROOT = Path(__file__).resolve().parents[2]
if str(_MMC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMC2_ROOT))

from sbi_forward_sim.src.physics_decoder import (  # noqa: E402
    decode_bip_batch,
    decode_bip_from_sample,
)
from sbi_forward_sim.src.physics_decoder_contract import (  # noqa: E402
    DECODER_G_NUMERIC_REQUIRED,
    DECODER_G_OPTIONAL,
    DecoderInputError,
    DecoderInputs,
    physics_decoder_input_manifest_dict,
    validate_decoder_inputs,
)

# One reproducible admissible synthetic draw (found via small random search).
_ADMISSIBLE_SYNTHETIC = dict(
    g={
        "vx0": -5.0,
        "vy0": -118.0,
        "vz0": -2.0,
        "ax": 8.0,
        "ay": 22.0,
        "az": -15.0,
        "release_pos_y": 54.5,
        "release_spin_rate": 2100.0,
        "spin_axis_sin": math.sin(math.radians(210)),
        "spin_axis_cos": math.cos(math.radians(210)),
    },
    p={
        "bat_length_in": 34.0,
        "bat_weight_oz": 32.0,
        "x_cm_fixed_in": 24.0,
        "r_g_fixed_in": 14.0,
        "I0_oz_in2": 32.0 * 14.0**2,
        "Iz_oz_in2": 1500.0,
    },
    u={"v_ss_tilde": 70.0, "a_tilde": -12.0, "d_tilde": 30.0},
    z={
        "x": 31.34442185152505,
        "psi_deg": 20.636352235224194,
        "e_y_star": 0.30514289520771126,
        "theta_deg": 14.062086260253718,
    },
)


_G_ACCESS_ALLOWED: frozenset[str] = (
    DECODER_G_NUMERIC_REQUIRED
    | DECODER_G_OPTIONAL
    | frozenset({"spin_axis_deg", "spin_axis_sin", "spin_axis_cos"})
)


def _decoder_g_subscript_keys(py_path: Path) -> set[str]:
    """Collect string keys read from context dict `g` or `inp.g` subscripts / .get(...)."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"))

    def slice_key(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    keys: set[str] = set()

    def is_context_g(node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id == "g":
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "g"
            and isinstance(node.value, ast.Name)
            and node.value.id == "inp"
        ):
            return True
        return False

    class V(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if is_context_g(node.value):
                k = slice_key(node.slice)
                if k is not None:
                    keys.add(k)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and is_context_g(node.func.value)
                and node.args
            ):
                k = slice_key(node.args[0])
                if k is not None:
                    keys.add(k)
            self.generic_visit(node)

    V().visit(tree)
    return keys


def _check_la_sa_tan(out, u: dict, z: dict, tol: float) -> list[str]:
    err: list[str] = []
    if not out.admissible:
        err.append("expected admissible synthetic row")
        return err
    if abs(out.launch_angle_deg - float(z["theta_deg"])) > tol:
        err.append(f"LA mismatch: {out.launch_angle_deg} vs {z['theta_deg']}")
    if abs(out.spray_angle_deg - float(u["d_tilde"])) > tol:
        err.append(f"SA mismatch: {out.spray_angle_deg} vs {u['d_tilde']}")
    vn = out.V_n_plus_fps
    vt = out.V_t_plus_fps
    cd = out.cos_theta_minus_psi
    sd = out.sin_theta_minus_psi
    if vn is None or vt is None or cd is None or sd is None:
        err.append("missing Vn/Vt or trig diagnostics")
        return err
    ratio = vt / vn
    expect = math.tan(math.radians(float(z["theta_deg"])) - math.radians(float(z["psi_deg"])))
    if abs(ratio - expect) > 1e-7:
        err.append(f"tan(theta-psi) mismatch: Vt/Vn={ratio} vs tan={expect}")
    mph_to_fps = 5280.0 / 3600.0
    s_fps = out.ev_mph * mph_to_fps
    if abs(vn - s_fps * cd) > 1e-4 or abs(vt - s_fps * sd) > 1e-4:
        err.append(f"Vn/Vt not consistent with s*cos/sin: Vn={vn} vt={vt} s_fps={s_fps}")
    return err


def _check_source_bans(decoder_path: Path) -> list[str]:
    err: list[str] = []
    text = decoder_path.read_text(encoding="utf-8")
    if re.search(r"math\.tan\(\s*psi", text) or "tan(psi - theta)" in text or "tan(psi_rad - theta" in text:
        err.append("suspected tan(psi - theta) pattern in decoder source")
    if "math.cos(psi_rel" in text or "math.sin(psi_rel" in text:
        err.append("suspected use of psi_rel as plane angle for nhat/that")
    return err


def _check_manifest_alignment(project_root: Path) -> list[str]:
    err: list[str] = []
    man_path = project_root / "manifests" / "physics_decoder_input_contract.json"
    if not man_path.is_file():
        err.append(f"missing {man_path}")
        return err
    on_disk = json.loads(man_path.read_text(encoding="utf-8"))
    live = physics_decoder_input_manifest_dict()
    if sorted(on_disk.get("g", {}).get("required_keys_always", [])) != sorted(
        live["g"]["required_keys_always"]
    ):
        err.append("manifest JSON g.required_keys_always out of sync; re-run write_physics_decoder_input_manifest()")
    if on_disk.get("g", {}).get("spin_axis") != live["g"]["spin_axis"]:
        err.append("manifest JSON g.spin_axis out of sync")
    return err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol-deg", type=float, default=1e-9)
    ap.add_argument(
        "--decoder-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "physics_decoder.py",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = ap.parse_args()
    errors: list[str] = []

    errors.extend(_check_source_bans(args.decoder_path))
    errors.extend(_check_manifest_alignment(args.project_root))

    g_keys = _decoder_g_subscript_keys(args.decoder_path)
    stray = sorted(g_keys - _G_ACCESS_ALLOWED)
    if stray:
        errors.append(
            "physics_decoder.py accesses g/inp.g keys not in locked contract allowlist: "
            + ", ".join(stray)
            + " (update DECODER_G_* in physics_decoder_contract.py and manifest, or remove stray access)."
        )

    inp = DecoderInputs(
        event_id=1,
        batter_name="synthetic",
        g=_ADMISSIBLE_SYNTHETIC["g"],
        p=_ADMISSIBLE_SYNTHETIC["p"],
        u=_ADMISSIBLE_SYNTHETIC["u"],
        z=_ADMISSIBLE_SYNTHETIC["z"],
    )
    validate_decoder_inputs(inp)

    # Missing required g must raise before any physics.
    g_bad = {k: v for k, v in _ADMISSIBLE_SYNTHETIC["g"].items() if k != "release_pos_y"}
    bad_inp = DecoderInputs(
        event_id=0,
        batter_name="synthetic",
        g=g_bad,
        p=_ADMISSIBLE_SYNTHETIC["p"],
        u=_ADMISSIBLE_SYNTHETIC["u"],
        z=_ADMISSIBLE_SYNTHETIC["z"],
    )
    try:
        decode_bip_from_sample(bad_inp)
        errors.append("expected DecoderInputError when release_pos_y missing")
    except DecoderInputError as ex:
        if "release_pos_y" not in str(ex):
            errors.append(f"missing release_pos_y error should name column; got: {ex}")

    g_no_spin = {
        k: v
        for k, v in _ADMISSIBLE_SYNTHETIC["g"].items()
        if k not in ("spin_axis_sin", "spin_axis_cos", "spin_axis_deg")
    }
    spinless = DecoderInputs(
        event_id=0,
        batter_name="synthetic",
        g=g_no_spin,
        p=_ADMISSIBLE_SYNTHETIC["p"],
        u=_ADMISSIBLE_SYNTHETIC["u"],
        z=_ADMISSIBLE_SYNTHETIC["z"],
    )
    try:
        decode_bip_from_sample(spinless)
        errors.append("expected DecoderInputError when spin axis missing")
    except DecoderInputError as ex:
        if "spin_axis" not in str(ex).lower():
            errors.append(f"spin error should mention spin_axis; got: {ex}")

    out = decode_bip_from_sample(inp)
    errors.extend(_check_la_sa_tan(out, _ADMISSIBLE_SYNTHETIC["u"], _ADMISSIBLE_SYNTHETIC["z"], args.tol_deg))

    # Inadmissible: psi outside support
    bad_z = dict(_ADMISSIBLE_SYNTHETIC["z"])
    bad_z["psi_deg"] = 85.0
    bad_inp2 = DecoderInputs(
        event_id=2,
        batter_name="synthetic",
        g=_ADMISSIBLE_SYNTHETIC["g"],
        p=_ADMISSIBLE_SYNTHETIC["p"],
        u=_ADMISSIBLE_SYNTHETIC["u"],
        z=bad_z,
    )
    bad_out = decode_bip_from_sample(bad_inp2)
    if bad_out.admissible or not bad_out.failure_reason:
        errors.append("expected failure_reason on psi out of support")
    if bad_out.failure_reason != "psi_outside_support":
        errors.append(f"unexpected failure_reason: {bad_out.failure_reason}")

    df = decode_bip_batch([inp, bad_inp2])
    if len(df) != 2 or "EV" not in df.columns:
        errors.append("batch frame malformed")

    if errors:
        print("FAILED:")
        for e in errors:
            print(" ", e)
        return 1
    print("OK: physics_decoder validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
