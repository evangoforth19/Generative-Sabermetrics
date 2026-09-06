"""Generate markdown audit reports for SBI preprocessing (tables + policy checks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_engineering import standardization_numeric_blocks
from .io_utils import read_table

ANGLE_SUBSTRINGS = (
    "launch",
    "spray",
    "phi",
    "theta",
    "attack",
    "d_tilde",
    "tilde",
)


def _inspect_parquet_or_csv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        return out
    df = read_table(path)
    out["row_count"] = int(len(df))
    out["columns"] = list(df.columns)
    out["angle_related"] = [
        c
        for c in df.columns
        if any(s in c.lower() for s in ANGLE_SUBSTRINGS)
    ]
    return out


def write_column_audit_theta_and_angles(production_root: Path, report_path: Path) -> None:
    # Prefer parquet when present; CSV only when parquet is absent (file-format policy).
    logical_specs: list[tuple[str, str | None]] = [
        ("selected_events", "selected_events.csv"),
        ("aggregates/context_event_table", None),
        ("aggregates/event_quality_flags", None),
        ("aggregates/chain_diagnostics", None),
        ("train_exports/posterior_draws_long_raw", None),
        ("train_exports/posterior_draws_long_basic_screened", None),
        ("train_exports/posterior_draws_long_strict_screened", None),
        ("train_exports/context_to_latent_train", "train_exports/context_to_latent_train.csv"),
        ("train_exports/context_to_upstream_train", "train_exports/context_to_upstream_train.csv"),
        (
            "train_exports/context_upstream_latent_joint_train",
            "train_exports/context_upstream_latent_joint_train.csv",
        ),
        ("train_exports/event_level_targets", None),
        ("calibration_exports/observed_inference_pack", None),
    ]
    lines = ["# Column audit: θ, launch angle, spray, and related fields", ""]
    lines.append(f"Production root: `{production_root}`")
    lines.append("")
    for base, csv_fallback in logical_specs:
        pq = production_root / f"{base}.parquet"
        if pq.is_file():
            rel_used = f"{base}.parquet"
            p = pq
        elif csv_fallback and (production_root / csv_fallback).is_file():
            rel_used = csv_fallback
            p = production_root / csv_fallback
        else:
            rel_used = f"{base}.parquet (missing; CSV fallback not used or absent)"
            p = pq
        info = _inspect_parquet_or_csv(p)
        info["logical_name"] = rel_used
        lines.append(f"## `{rel_used}`")
        lines.append(f"- path: `{info['path']}`")
        lines.append(f"- exists: {info['exists']}")
        if info.get("row_count") is not None:
            lines.append(f"- row count: {info['row_count']:,}")
            lines.append(f"- column count: {len(info['columns'])}")
            lines.append("- all columns:")
            for c in info["columns"]:
                lines.append(f"  - `{c}`")
            ar = info.get("angle_related") or []
            lines.append("- angle / spray / θ-related subset:")
            if ar:
                for c in ar:
                    lines.append(f"  - `{c}`")
            else:
                lines.append("  - _(none detected by substring filter)_")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_fair_wedge_audit(
    production_root: Path,
    manifest_extra: dict[str, Any],
    report_path: Path,
) -> None:
    extra = dict(manifest_extra)
    ctx_path = production_root / "aggregates/context_event_table.parquet"
    if ctx_path.is_file() and "context_rows_before_bip_filter" not in extra:
        try:
            from .dataset_builders import bip_model_training_event_mask

            ctx = read_table(ctx_path)
            extra["context_rows_before_bip_filter"] = int(len(ctx))
            extra["context_rows_after_bip_filter"] = int(bip_model_training_event_mask(ctx).sum())
        except Exception as exc:  # noqa: BLE001
            extra["fair_wedge_rows_error"] = repr(exc)

    lines = [
        "# Fair wedge and physical plausibility audit",
        "",
        "Filtering matches `run_mcmc_posterior_bank` **bip_model** base mask plus **fair wedge** "
        "(`~flag_outside_fair_wedge`), using only `aggregates/context_event_table.parquet` flag columns.",
        "",
        "## Row counts (context / events)",
        "",
        f"- context rows before bip_model + fair-wedge filter: **{extra.get('context_rows_before_bip_filter', 'n/a')}**",
        f"- context rows after filter: **{extra.get('context_rows_after_bip_filter', 'n/a')}**",
        "",
    ]
    for k in sorted(extra.keys()):
        if "draw_rows" in k and ("bip" in k or "final" in k):
            lines.append(f"- `{k}`: **{extra[k]}**")
    lines.extend(
        [
            "",
            "## Flag columns required on context master",
            "",
        ]
    )
    bip_flags = [
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
    ]
    for c in bip_flags + ["flag_outside_fair_wedge"]:
        lines.append(f"- `{c}`: must be present; bip_model keeps rows where bad flags are **false** and outside-fair-wedge is **false**.")
    lines.extend(
        [
            "",
            "## Rebuilt datasets",
            "",
            "After filter, all stage **u**, **z**, **joint**, and **baseline_direct_y** parquet outputs are built only from eligible `event_id` rows.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_theta_phi_semantic_audit(project_root: Path, report_path: Path) -> None:
    proc = project_root / "data_processed"
    lines = [
        "# θ vs φ / spray semantic audit (processed outputs)",
        "",
        "Checks processed artifacts for ambiguous or incorrect angle semantics.",
        "",
    ]
    if not proc.is_dir():
        lines.append("_data_processed/ missing — run build first._")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for name in ("u_train", "z_train", "joint_train"):
        fp = proc / f"{name}.parquet"
        if not fp.is_file():
            lines.append(f"## `{name}`")
            lines.append("_file missing_")
            lines.append("")
            continue
        df = pd.read_parquet(fp)
        cols = list(df.columns)
        lines.append(f"## `{name}` ({len(df):,} rows)")
        lines.append(f"- columns: {', '.join(cols)}")
        contamination = [c for c in cols if c == "theta"]
        if name == "z_train":
            from .schema import Z_TARGET_COLUMNS

            miss = [c for c in Z_TARGET_COLUMNS if c not in cols]
            if miss:
                lines.append(f"- **ERROR**: z_train missing stage-z targets: {miss}")
            if "phi_star" in cols:
                lines.append(
                    "- **WARNING**: `phi_star` present on z slice; must not be used as launch angle."
                )
            if "theta_deg" not in cols:
                lines.append("- **ERROR**: z_train missing canonical `theta_deg`")
        if name == "u_train" and "phi_star" in cols:
            lines.append("- **ERROR**: `phi_star` must not appear on u_train.")
        if contamination:
            lines.append(f"- **potential ambiguity columns**: {contamination}")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_standardization_validation(project_root: Path, report_path: Path) -> None:
    proc = project_root / "data_processed"
    man = project_root / "manifests"
    stats_path = man / "standardization_stats.json"
    lines = [
        "# Standardization validation",
        "",
    ]
    if not stats_path.is_file():
        lines.append("_standardization_stats.json not found; run build first._")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    draws = pd.read_parquet(proc / "sbi_draws_basic.parquet")
    split = pd.read_parquet(proc / "event_split_table.parquet")
    merged = draws.copy()
    merged["split"] = merged["event_id"].map(split.set_index("event_id")["split"])
    if merged["split"].isna().any():
        n = int(merged["split"].isna().sum())
        lines.append(f"- **WARNING**: {n} draw rows could not be mapped to `event_split_table.split`; check stale artifacts.")
        merged = merged.loc[merged["split"].notna()]
    train = merged.loc[merged["split"].eq("train")]
    num_cols = standardization_numeric_blocks(merged)
    forbidden = {"event_id", "split", "batter_name"}
    problems: list[str] = []
    for c in stats:
        if c in forbidden:
            problems.append(f"stats unexpectedly include forbidden id column `{c}`")
    for c in stats:
        if c not in train.columns:
            problems.append(f"column `{c}` in stats but not in training draws")
            continue
        s = pd.to_numeric(train[c], errors="coerce")
        mu = float(s.mean(skipna=True))
        sig = float(s.std(skipna=True))
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1.0
        sm = stats[c]
        if not np.isclose(mu, sm["mean"], rtol=0, atol=1e-4):
            problems.append(f"`{c}` mean mismatch: recompute {mu} vs stored {sm['mean']}")
        if not np.isclose(sig, sm["std"], rtol=1e-5, atol=1e-4):
            problems.append(f"`{c}` std mismatch: recompute {sig} vs stored {sm['std']}")

    lines.append(f"- Standardized columns ({len(stats)}): {', '.join(sorted(stats.keys()))}")
    lines.append(f"- Candidate numeric schema columns considered: {', '.join(num_cols)}")
    lines.append(f"- Train draw rows used for check: {len(train):,}")
    if problems:
        lines.append("")
        lines.append("## Issues")
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("")
        lines.append("## Result")
        lines.append("- Train-only means/stds in `standardization_stats.json` match recomputation on `sbi_draws_basic` train split.")
        lines.append("- Identifiers and `split` are not in the stats file (only g/p/u/z numeric blocks).")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_all_audit_reports(
    production_root: Path,
    project_root: Path,
    manifest_extra: dict[str, Any] | None = None,
) -> None:
    rep = project_root / "reports"
    manifest_extra = manifest_extra or {}
    write_column_audit_theta_and_angles(production_root, rep / "column_audit_theta_and_angles.md")
    write_fair_wedge_audit(
        production_root,
        manifest_extra,
        rep / "fair_wedge_and_physical_plausibility_audit.md",
    )
    write_theta_phi_semantic_audit(project_root, rep / "theta_phi_semantic_audit.md")
    write_standardization_validation(project_root, rep / "standardization_validation.md")
