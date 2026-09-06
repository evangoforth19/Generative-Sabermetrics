"""
Deterministic physics decoder contract (exact-root inverse-collision aligned).

Concrete implementation: `sbi_forward_sim.src.physics_decoder` (`decode_bip_from_sample`,
`decode_bip_batch`, `ExactRootPhysicsDecoder`). Implementations must stay aligned with
`run_inverse_collision_production.py` admissibility and the exact-root production lineage.

------------------------------------------------------------------------------
Neural vs decoder context (**g**)
------------------------------------------------------------------------------
Frozen **stage-u** and **stage-z** models consume the feature sets described in
`schema.G_COLUMNS` (and configs); those contracts are unchanged.

The **physics decoder** uses a **stricter context dict `g`**: every key listed under
`DECODER_G_NUMERIC_REQUIRED` plus a **spin-axis bundle** (see `DECODER_G_SPIN_RULE`)
must be present and finite. In particular **`release_pos_y` is required** for
production-consistent contact-time and incoming-velocity reconstruction
(`build_incoming_pitch_from_sensor`), matching `run_mcmc_posterior_bank`.

The full public requirement set is machine-readable in
`manifests/physics_decoder_input_contract.json` and enforced by `validate_decoder_inputs`
(raises `DecoderInputError` if violated).

------------------------------------------------------------------------------
Identifiers, **u**, **z**, **p**
------------------------------------------------------------------------------
- **u:** stage-u outputs `v_ss_tilde`, `a_tilde`, `d_tilde` (tilde calibration lineage;
  `d_tilde` is spray angle φ in degrees).
- **z:** either **legacy** ``{x, psi_deg, e_y_star, theta_deg}`` (θ sampled) or **ex_analytic**
  ``{x, psi_deg, e_y_star, e_x}`` (tangential restitution sampled; θ and EV from the
  closed-form post-collision map inside the decoder). Angles in degrees in `{r_hat(phi), e_z}`.
- **p (physics subset):** keys actually read when constructing `Hitter` in the decoder
  (`DECODER_P_REQUIRED`). `schema.P_COLUMNS` still lists `I0_oz_in2` for neural/export
  row shape; the decoder **does not read** `I0_oz_in2` because `Hitter` sets
  `I0_oz_in2 = bat_weight_oz * r_g_fixed**2` internally.

``omega_plus`` is not a stage-z neural input; it remains a downstream diagnostic when exported.

**Outputs:** observables **y** = (EV, LA, SA) in physical units, plus optional diagnostics.

**Hard admissibility** (non-exhaustive; production-aligned): see implementation report.

Types use float64 where applicable; angles in degrees unless noted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

# --- Public required sets (minimal columns the decoder reads) -----------------

DECODER_U_REQUIRED: Final[frozenset[str]] = frozenset(
    {"v_ss_tilde", "a_tilde", "d_tilde"}
)

# Stage-z may supply either legacy (theta sampled) or ex-analytic (e_x sampled; theta from decoder).
DECODER_Z_LEGACY: Final[frozenset[str]] = frozenset({"x", "psi_deg", "e_y_star", "theta_deg"})
DECODER_Z_EX_ANALYTIC: Final[frozenset[str]] = frozenset({"x", "psi_deg", "e_y_star", "e_x"})
DECODER_Z_REQUIRED: Final[frozenset[str]] = DECODER_Z_LEGACY  # backward-compat alias for docs

# Always-required numeric/string keys on g (production pitch reconstruction path).
# Spin axis is validated separately (two allowed bundles) — see DECODER_G_SPIN_RULE.
DECODER_G_NUMERIC_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "release_pos_y",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "release_spin_rate",
    }
)

DECODER_G_SPIN_RULE: Final[str] = (
    "Provide finite `spin_axis_deg` OR both finite `spin_axis_sin` and "
    "`spin_axis_cos` (2D proxy consistent with processed SBI tables and bank omega_minus)."
)

DECODER_G_OPTIONAL: Final[frozenset[str]] = frozenset({"active_spin_ratio"})

# Subset of schema.P_COLUMNS read by `hitter_from_p_dict` in physics_decoder.py.
DECODER_P_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "bat_length_in",
        "bat_weight_oz",
        "x_cm_fixed_in",
        "r_g_fixed_in",
        "Iz_oz_in2",
    }
)


class DecoderInputError(ValueError):
    """Public decoder inputs violate the locked contract (missing or non-finite field)."""


def _finite_scalar(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def decoder_z_branch(z: dict[str, Any]) -> Literal["ex_analytic", "legacy"]:
    """``ex_analytic`` when ``e_x`` is provided (and finite) and ``theta_deg`` is not used."""
    has_ex = "e_x" in z and _finite_scalar(z["e_x"])
    has_th = "theta_deg" in z and _finite_scalar(z.get("theta_deg"))
    if has_ex and not has_th:
        return "ex_analytic"
    return "legacy"


def validate_decoder_context_g(g: dict[str, Any]) -> None:
    """
    Validate decoder **g** (context) only: always-required keys + spin-axis rule.
    Optional: `active_spin_ratio` (else bank default at runtime).
    """
    missing = sorted(k for k in DECODER_G_NUMERIC_REQUIRED if k not in g)
    if missing:
        raise DecoderInputError(
            "Decoder context g missing required keys: "
            + ", ".join(missing)
            + ". Neural feature lists (schema.G_COLUMNS) alone are insufficient; "
            "see manifests/physics_decoder_input_contract.json."
        )
    bad = [k for k in DECODER_G_NUMERIC_REQUIRED if not _finite_scalar(g[k])]
    if bad:
        raise DecoderInputError(
            "Decoder context g has non-finite or non-numeric required fields: "
            + ", ".join(sorted(bad))
        )

    has_deg = "spin_axis_deg" in g and _finite_scalar(g["spin_axis_deg"])
    has_trig = (
        "spin_axis_sin" in g
        and "spin_axis_cos" in g
        and _finite_scalar(g["spin_axis_sin"])
        and _finite_scalar(g["spin_axis_cos"])
    )
    if not has_deg and not has_trig:
        raise DecoderInputError(
            f"Decoder context g: {DECODER_G_SPIN_RULE} "
            "(see physics_decoder_contract.DECODER_G_SPIN_RULE)."
        )

    extra = set(g.keys()) - DECODER_G_NUMERIC_REQUIRED - DECODER_G_OPTIONAL - {
        "spin_axis_deg",
        "spin_axis_sin",
        "spin_axis_cos",
    }
    # Do not error on extra keys — pipelines may merge wide tables; only required keys matter.


def validate_decoder_inputs(inp: "DecoderInputs") -> None:
    """
    Strict public gate: fail loudly before loading production bank modules or doing physics.

    Raises:
        DecoderInputError: missing identifier, u/z/g/p keys, or non-finite required scalars.
    """
    errs: list[str] = []

    if not isinstance(inp.batter_name, str) or not inp.batter_name.strip():
        errs.append("batter_name must be a non-empty string")

    try:
        int(inp.event_id)
    except (TypeError, ValueError):
        errs.append("event_id must be int-convertible")

    for label, block, required in (
        ("u", inp.u, DECODER_U_REQUIRED),
        ("p", inp.p, DECODER_P_REQUIRED),
    ):
        missing = sorted(k for k in required if k not in block)
        if missing:
            errs.append(f"{label} missing required keys: {missing}")
        else:
            bad = [k for k in required if not _finite_scalar(block[k])]
            if bad:
                errs.append(f"{label} has non-finite required fields: {sorted(bad)}")

    zreq = DECODER_Z_EX_ANALYTIC if decoder_z_branch(inp.z) == "ex_analytic" else DECODER_Z_LEGACY
    missing_z = sorted(k for k in zreq if k not in inp.z)
    if missing_z:
        errs.append(f"z missing required keys for {decoder_z_branch(inp.z)} mode: {missing_z}")
    else:
        bad_z = [k for k in zreq if not _finite_scalar(inp.z[k])]
        if bad_z:
            errs.append(f"z has non-finite required fields: {sorted(bad_z)}")

    if errs:
        raise DecoderInputError("; ".join(errs))

    validate_decoder_context_g(inp.g)


def physics_decoder_input_manifest_dict() -> dict[str, Any]:
    """Single dict for JSON manifest and docs (versioned)."""
    return {
        "manifest_version": 1,
        "description": (
            "Minimal public inputs for sbi_forward_sim physics decoder "
            "(stricter than neural-only schema.G_COLUMNS)."
        ),
        "identifiers": {
            "event_id": {"required": True, "type": "int", "note": "Book-keeping / batch joins."},
            "batter_name": {"required": True, "type": "str", "note": "Player id for p/hitter join."},
        },
        "u": {
            "source": "stage_u_outputs",
            "required_keys": sorted(DECODER_U_REQUIRED),
            "units": {
                "v_ss_tilde": "mph (calibrated sweet-spot bat speed lineage)",
                "a_tilde": "deg (attack angle, calib-to-spray lineage)",
                "d_tilde": "deg (spray direction phi, calib lineage)",
            },
        },
        "z": {
            "source": "stage_z_outputs",
            "branch_rule": "decoder_z_branch(z): ex_analytic if finite e_x and no finite theta_deg; else legacy.",
            "legacy_required_keys": sorted(DECODER_Z_LEGACY),
            "ex_analytic_required_keys": sorted(DECODER_Z_EX_ANALYTIC),
            "required_keys_note": "DECODER_Z_REQUIRED in code is the legacy set for backward-compat imports only.",
            "units": {
                "x": "in (contact distance from knob along bat)",
                "psi_deg": "deg (contact normal tilt in {r_hat(phi), e_z} plane)",
                "e_y_star": "unitless (normal-direction restitution latent e_y*)",
                "theta_deg": "deg (legacy only: outgoing polar angle in that plane)",
                "e_x": "unitless [0, 0.6] (ex_analytic only: tangential restitution latent)",
            },
        },
        "g": {
            "source": "event_context",
            "required_keys_always": sorted(DECODER_G_NUMERIC_REQUIRED),
            "spin_axis": DECODER_G_SPIN_RULE,
            "optional_keys": sorted(DECODER_G_OPTIONAL),
            "notes": {
                "release_pos_y": (
                    "feet; Statcast release extension along y — REQUIRED for decoder parity "
                    "with run_mcmc_posterior_bank.build_incoming_pitch_from_sensor (contact time)."
                ),
                "release_spin_rate": "rpm (feeds relevant_omega_minus_from_sensor).",
                "vx0_vy0_vz0_ax_ay_az": (
                    "Statcast pitch kinematics at pitch release (ft/s and ft/s^2 convention per bank)."
                ),
            },
            "neural_g_columns_note": (
                "schema.G_COLUMNS is the frozen neural feature list and does NOT include "
                "release_pos_y; joins for full forward simulation must supply decoder extras."
            ),
        },
        "p": {
            "source": "player_constants",
            "required_keys_decoder_physics": sorted(DECODER_P_REQUIRED),
            "schema_p_columns_note": (
                "schema.P_COLUMNS also lists I0_oz_in2 for training row consistency; "
                "decoder physics does not read it (Hitter derives I0 from weight and r_g)."
            ),
            "units": {
                "bat_length_in": "in",
                "bat_weight_oz": "oz",
                "x_cm_fixed_in": "in",
                "r_g_fixed_in": "in",
                "Iz_oz_in2": "oz*in^2",
            },
        },
    }


def default_physics_decoder_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "manifests" / "physics_decoder_input_contract.json"


def write_physics_decoder_input_manifest(path: Path | None = None) -> Path:
    """Write `physics_decoder_input_manifest_dict()` to JSON (stable keys)."""
    path = path or default_physics_decoder_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(physics_decoder_input_manifest_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass
class DecoderInputs:
    """Bundled inputs for a single posterior draw or simulator sample."""

    event_id: int
    batter_name: str
    g: dict[str, Any]
    p: dict[str, Any]
    u: dict[str, float]
    z: dict[str, float]


@dataclass
class DecoderOutputs:
    """Per-sample decode result. Extended fields are None / NaN when inadmissible."""

    ev_mph: float
    launch_angle_deg: float
    spray_angle_deg: float
    e_x: float | None
    omega_plus_rad_s: float | None
    admissible: bool
    failure_reason: str | None
    # Optional diagnostics (fps for collision-frame speeds; production uses fps internally)
    V_n_plus_fps: float | None = None
    V_t_plus_fps: float | None = None
    cos_theta_minus_psi: float | None = None
    sin_theta_minus_psi: float | None = None
    log_prior_ex: float | None = None
    regime_label: str | None = None
    lambda_xy: float | None = None
    r_x_x: float | None = None
    r_y_x: float | None = None
    psi_rel_legacy_deg: float | None = None
    d_n: float | None = None
    d_t: float | None = None
    x_in: float | None = None
    # Lenient mode: inverse-style gates treated as diagnostics only; semicolon-separated.
    strict_diagnostic_flags: str | None = None


class PhysicsDecoder(Protocol):
    """Deterministic decoder implementing production-consistent mapping (u, z, g, p) -> (EV, LA, SA)."""

    def decode(self, inp: DecoderInputs) -> DecoderOutputs:
        ...
