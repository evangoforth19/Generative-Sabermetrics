"""
Build frozen stage-u / stage-z encoder inputs from baseline + context master rows.

Used for held-out forward simulation without requiring membership in u_test / z_test parquets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_u import encode_categories
from .physics_decoder_contract import (
    DECODER_P_REQUIRED,
    DecoderInputError,
    validate_decoder_context_g,
)


U_PLACEHOLDER_TARGETS = ("v_ss_tilde", "a_tilde", "d_tilde")


def merge_baseline_master_row(brow: pd.Series, master: pd.DataFrame) -> pd.Series:
    """Baseline row with master-filled NaNs and any master-only columns needed downstream."""
    eid = int(brow["event_id"])
    msub = master.loc[master["event_id"] == eid]
    if len(msub) == 0:
        raise DecoderInputError(f"event_id {eid} missing from context master")
    if len(msub) > 1:
        raise DecoderInputError(f"event_id {eid} duplicate rows in context master ({len(msub)})")
    mrow = msub.iloc[0]
    out = brow.copy()
    for c in mrow.index:
        if c == "event_id":
            continue
        if c not in out.index:
            out[c] = mrow[c]
        elif pd.isna(out.get(c)):
            out[c] = mrow[c]
    return out


def validate_decoder_row(mrow: pd.Series) -> None:
    from .physics_decoder_contract import DECODER_G_NUMERIC_REQUIRED, DECODER_G_OPTIONAL

    g: dict[str, Any] = {}
    for k in DECODER_G_NUMERIC_REQUIRED:
        if k not in mrow.index:
            raise DecoderInputError(f"decoder g missing {k!r}")
        g[k] = float(mrow[k])
    has_deg = "spin_axis_deg" in mrow.index and pd.notna(mrow.get("spin_axis_deg"))
    if has_deg:
        g["spin_axis_deg"] = float(mrow["spin_axis_deg"])
    else:
        for k in ("spin_axis_sin", "spin_axis_cos"):
            g[k] = float(mrow[k])
    for ok in DECODER_G_OPTIONAL:
        if ok in mrow.index and pd.notna(mrow.get(ok)):
            g[ok] = float(mrow[ok])
    validate_decoder_context_g(g)
    for k in DECODER_P_REQUIRED:
        if k not in mrow.index or pd.isna(mrow.get(k)):
            raise DecoderInputError(f"decoder p missing or NaN {k!r}")


def u_encoder_inputs_from_row(
    row: pd.Series,
    stats: dict[str, dict[str, float]],
    vocabs: dict[str, dict[str, int]],
    u_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return x_u (1, F) standardized and categorical ints (1,) per key (sorted keys)."""
    num_cols = list(u_cfg["numeric_features"])
    pcols = list(u_cfg["player_constant_features"])
    x_parts = []
    for c in num_cols + pcols:
        if c not in stats:
            raise KeyError(f"standardization stats missing {c!r}")
        v = float(pd.to_numeric(row[c], errors="coerce"))
        if not np.isfinite(v):
            raise ValueError(f"non-finite u feature {c!r} for event {row.get('event_id')}")
        mu, sig = float(stats[c]["mean"]), float(stats[c]["std"])
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        x_parts.append((v - mu) / sig)
    x_num = np.array([x_parts], dtype=np.float32)

    df1 = pd.DataFrame([row.to_dict()])
    cat: dict[str, np.ndarray] = {}
    for col in sorted(vocabs.keys()):
        cat[col] = encode_categories(df1, col, vocabs[col])
    return x_num, cat


def z_base_x_num_from_row(
    row: pd.Series,
    num_order: list[str],
    stats: dict[str, dict[str, float]],
    u_cols: tuple[str, str, str],
) -> np.ndarray:
    """
    Standardized z neural inputs (F,), with dummy 0 for upstream u slots (overwritten after u sampling).
    """
    out = np.zeros(len(num_order), dtype=np.float32)
    for i, c in enumerate(num_order):
        if c in u_cols:
            out[i] = 0.0
            continue
        if c not in stats:
            raise KeyError(f"standardization stats missing {c!r}")
        v = float(pd.to_numeric(row[c], errors="coerce"))
        if not np.isfinite(v):
            raise ValueError(f"non-finite z context feature {c!r} for event {row.get('event_id')}")
        mu, sig = float(stats[c]["mean"]), float(stats[c]["std"])
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        out[i] = (v - mu) / sig
    return out


def z_categoricals_from_row(row: pd.Series, vocabs: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    df1 = pd.DataFrame([row.to_dict()])
    return {col: encode_categories(df1, col, vocabs[col]) for col in sorted(vocabs.keys())}


def build_corrected_eligibility_audit(
    baseline_test: pd.DataFrame,
    master: pd.DataFrame,
    stats_u: dict[str, dict[str, float]],
    stats_z: dict[str, dict[str, float]],
    u_cfg: dict[str, Any],
    z_num_order: list[str],
    u_vocabs: dict[str, dict[str, int]],
    z_vocabs: dict[str, dict[str, int]],
    u_cols: tuple[str, str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eligible = baseline test row with y + decoder-complete master + encodable u/z features."""
    rows: list[dict[str, Any]] = []
    n_y = n_dec = n_u = n_z = n_elig = 0
    bl_by = baseline_test.set_index("event_id", drop=False)
    for eid, brow in bl_by.iterrows():
        eid = int(eid)
        ok_y = all(t in brow.index and pd.notna(brow[t]) for t in ("EV", "LA", "SA"))
        if ok_y:
            n_y += 1
        msub = master.loc[master["event_id"] == eid]
        ok_dec = False
        dec_reason = ""
        if len(msub) == 0:
            dec_reason = "missing_context_master"
        elif len(msub) > 1:
            dec_reason = "duplicate_context_master"
        else:
            try:
                validate_decoder_row(msub.iloc[0])
                ok_dec = True
                n_dec += 1
            except (DecoderInputError, ValueError, TypeError) as ex:
                dec_reason = f"decoder_context:{ex}"

        ok_enc_u = False
        u_err = ""
        ok_enc_z = False
        z_err = ""
        try:
            row_m = merge_baseline_master_row(brow, master)
            u_encoder_inputs_from_row(row_m, stats_u, u_vocabs, u_cfg)
            ok_enc_u = True
            n_u += 1
        except Exception as ex:
            u_err = f"u_encode:{ex}"
        try:
            row_m = merge_baseline_master_row(brow, master)
            z_base_x_num_from_row(row_m, z_num_order, stats_z, u_cols)
            z_categoricals_from_row(row_m, z_vocabs)
            ok_enc_z = True
            n_z += 1
        except Exception as ex:
            z_err = f"z_encode:{ex}"

        eligible = ok_y and ok_dec and ok_enc_u and ok_enc_z
        if eligible:
            n_elig += 1
        excl: list[str] = []
        if not ok_y:
            excl.append("missing_or_nan_EV_LA_SA")
        if not ok_dec:
            excl.append(dec_reason or "decoder_incomplete")
        if not ok_enc_u:
            excl.append(u_err or "u_not_encodable")
        if not ok_enc_z:
            excl.append(z_err or "z_not_encodable")
        rows.append(
            {
                "event_id": eid,
                "has_observed_y": ok_y,
                "decoder_context_ok": ok_dec,
                "stage_u_encodable": ok_enc_u,
                "stage_z_encodable": ok_enc_z,
                "eligible_for_pipeline": eligible,
                "exclusion_reason": "" if eligible else "; ".join(excl),
            }
        )
    audit_df = pd.DataFrame(rows).sort_values("event_id")
    summary = {
        "n_baseline_direct_y_holdout_test_events": int(len(baseline_test)),
        "n_with_observed_EV_LA_SA": n_y,
        "n_decoder_context_complete_single_row": n_dec,
        "n_stage_u_encodable": n_u,
        "n_stage_z_encodable": n_z,
        "n_eligible_for_forward_pipeline": n_elig,
        "definition": "baseline_direct_y_test (official test split) ∩ decoder master ∩ u/z frozen encoders; no u_test/z_test parquet required",
    }
    inel = audit_df.loc[~audit_df["eligible_for_pipeline"], "exclusion_reason"]
    summary["exclusion_reason_counts_ineligible_top"] = (
        inel.value_counts().head(40).astype(int).to_dict() if len(inel) else {}
    )
    return audit_df, summary
