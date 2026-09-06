"""Explicit modeling blocks g, p, u, z, y and canonical column names."""

from __future__ import annotations

# Context g (pitch / plate appearance, pre-collision) — **neural / tabular modeling** only.
# The deterministic physics decoder requires additional context (notably `release_pos_y`);
# see `manifests/physics_decoder_input_contract.json` and `physics_decoder_contract`.
G_COLUMNS = [
    "z_count",
    "release_speed",
    "release_spin_rate",
    "spin_axis_sin",
    "spin_axis_cos",
    "plate_x",
    "plate_z",
    "pitch_type",
    "stand",
    "p_throws",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
]

# Player exogenous constants p
P_COLUMNS = [
    "bat_length_in",
    "bat_weight_oz",
    "x_cm_fixed_in",
    "r_g_fixed_in",
    "I0_oz_in2",
    "Iz_oz_in2",
]

# Transformed upstream u (stage 1 targets)
U_COLUMNS = ["v_ss_tilde", "a_tilde", "d_tilde"]

# SBI stage-z reduced target slice (forward simulator stage 2).
# theta_deg is canonical on processed tables: per draw, finite theta_star_deg else theta_obs_deg.
# e_x and omega_plus stay in production/joint draws only; solved inside the physics decoder, not stage-z targets.
Z_COLUMNS_CORE = ["x", "psi_deg", "e_y_star"]
Z_TARGET_COLUMNS = ["x", "psi_deg", "e_y_star", "theta_deg"]

# Source columns for building theta_deg (never phi_star / spray)
Z_THETA_SAMPLED = "theta_star_deg"
Z_THETA_OBSERVED = "theta_obs_deg"

# Observed BIP outputs y
Y_COLUMNS = ["EV", "LA", "SA"]

SELECTED_EVENTS_BAT_RENAME = {
    "L_in": "bat_length_in",
    "W_oz": "bat_weight_oz",
}


def modeling_schema_manifest(z_theta_provenance: str, *, z_theta_stats: dict | None = None) -> dict:
    """Authoritative modeling manifest for SBI; z is always the reduced stage-z slice."""
    out: dict = {
        "physics_decoder_input_manifest": "manifests/physics_decoder_input_contract.json",
        "g": G_COLUMNS,
        "p": P_COLUMNS,
        "u": U_COLUMNS,
        "z": list(Z_TARGET_COLUMNS),
        "z_theta_column": "theta_deg",
        "z_theta_provenance": z_theta_provenance,
        "production_latent_note": (
            "Broader posterior-bank exports may include e_x, omega_plus, etc. on long draws; "
            "the SBI forward simulator stage-z target vector is only Z_TARGET_COLUMNS; "
            "nuisance collision quantities are implied by the deterministic decoder."
        ),
        "decoder_context_g_note": (
            "Frozen neural models use `g` = G_COLUMNS. The physics decoder enforces a stricter "
            "event context dict (e.g. requires release_pos_y for bank contact-time parity); "
            "see manifests/physics_decoder_input_contract.json. Do not assume G_COLUMNS alone "
            "suffices for decode_bip_from_sample."
        ),
        "y": Y_COLUMNS,
        "notes": {
            "release_speed": (
                "True Statcast mph merged from batter_data pickle on "
                "(game_date, batter, pitcher, pitch_number); no ||v|| proxy."
            ),
            "theta_deg": (
                "Canonical stage-z launch angle: per draw, finite theta_star_deg else theta_obs_deg; "
                "never phi_star."
            ),
            "psi_deg": "Canonical name; sourced from psi_deg or psi (if degrees) in production draws.",
        },
    }
    if z_theta_stats:
        out["z_theta_row_stats"] = z_theta_stats
    return out


AVAILABLE_G_FROM_CONTEXT = [
    "pitch_type",
    "plate_x",
    "plate_z",
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
    "z_count",
]
