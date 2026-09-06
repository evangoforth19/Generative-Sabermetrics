#!/usr/bin/env python3
"""
Robustness metrics for a marginalized value-function distribution V
(e.g. xwOBAcon_3D = P(Y|EV,LA,SA) · w) against league BIP thresholds tau.

Either pass a single table (--input-path) or stack per-hitter bootstrap outputs
(--stack-run-root) so each hitter folder becomes a group (default column ``hitter_id``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

TAU_DEFAULTS: dict[str, float] = {
    "tau_league_mean": 0.582912,
    "tau_league_median": 0.473825,
    "tau_league_p25": 0.195527,
    "tau_league_p10": 0.023540,
}

VALUE_COL_CANDIDATES = ("xwOBAcon_3D", "xwobacon", "PW", "value", "V")


def load_table(path: Path) -> pd.DataFrame:
    p = path.resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p, low_memory=False)
    if suf in (".feather", ".fea"):
        return pd.read_feather(p)
    raise ValueError(f"Unsupported format: {p}")


def stack_value_tables_from_run_root(
    root: Path,
    *,
    filename: str = "simulated_ev_la_sa_value.parquet",
    group_col_name: str = "hitter_id",
) -> tuple[pd.DataFrame, int]:
    """Concatenate ``<root>/<hitter>/<filename>`` tables; tag rows with ``group_col_name`` = subdir name."""
    root = root.resolve()
    paths = sorted(root.glob(f"*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No `{filename}` files found under {root} (expected `<root>/*/{filename}`).")
    parts: list[pd.DataFrame] = []
    for p in paths:
        sub = load_table(p)
        sub[group_col_name] = p.parent.name
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    return df, len(paths)


def infer_value_col(df: pd.DataFrame, value_col: str | None) -> str:
    if value_col is not None:
        if value_col not in df.columns:
            raise ValueError(f"--value-col {value_col!r} not in dataframe columns.")
        return value_col
    for c in VALUE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer value column. Tried {list(VALUE_COL_CANDIDATES)}. "
        "Pass --value-col explicitly."
    )


def safe_divide(num: float, den: float) -> float:
    if not np.isfinite(den) or den == 0.0:
        if num > 0.0:
            return float("inf")
        if num == 0.0:
            return float("nan")
        if num < 0.0:
            return float("-inf")
    return float(num / den)


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = int(len(v))
    if n == 0:
        return {
            "n": 0,
            "mean_value": float("nan"),
            "std_value": float("nan"),
            "var_value": float("nan"),
            "min_value": float("nan"),
            "p01_value": float("nan"),
            "p05_value": float("nan"),
            "p10_value": float("nan"),
            "p25_value": float("nan"),
            "median_value": float("nan"),
            "p75_value": float("nan"),
            "p90_value": float("nan"),
            "p95_value": float("nan"),
            "p99_value": float("nan"),
            "max_value": float("nan"),
        }
    qs = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    pct = {q: float(np.percentile(v, q)) for q in qs}
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=0))
    return {
        "n": n,
        "mean_value": mu,
        "std_value": sd,
        "var_value": float(sd**2),
        "min_value": float(np.min(v)),
        "p01_value": pct[1],
        "p05_value": pct[5],
        "p10_value": pct[10],
        "p25_value": pct[25],
        "median_value": pct[50],
        "p75_value": pct[75],
        "p90_value": pct[90],
        "p95_value": pct[95],
        "p99_value": pct[99],
        "max_value": float(np.max(v)),
    }


def tau_metrics(values: np.ndarray, tau_name: str, tau_value: float) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    mu_v = float(np.mean(v)) if len(v) else float("nan")
    std_v = float(np.std(v, ddof=0)) if len(v) else float("nan")
    downside = np.maximum(tau_value - v, 0.0)
    upside = np.maximum(v - tau_value, 0.0)
    ed = float(np.mean(downside)) if len(v) else float("nan")
    eu = float(np.mean(upside)) if len(v) else float("nan")
    dd = float(np.sqrt(np.mean(downside**2))) if len(v) else float("nan")
    sortino = safe_divide(mu_v - tau_value, dd)
    omega_ud = safe_divide(eu, ed)
    omega_du = safe_divide(ed, eu)
    p_down = float(np.mean(v < tau_value)) if len(v) else float("nan")
    p_up = float(np.mean(v > tau_value)) if len(v) else float("nan")
    return {
        "tau_name": tau_name,
        "tau_value": float(tau_value),
        "mean_value": mu_v,
        "std_value": std_v,
        "downside_deviation": dd,
        "sortino_downside_ratio": sortino,
        "omega_up_down": omega_ud,
        "omega_down_up": omega_du,
        "expected_upside": eu,
        "expected_downside": ed,
        "downside_probability": p_down,
        "upside_probability": p_up,
        "mean_value_minus_tau": float(mu_v - tau_value) if np.isfinite(mu_v) else float("nan"),
    }


def compute_pooled_metrics(
    values: np.ndarray,
    taus: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dist = distribution_summary(values)
    tau_rows = [tau_metrics(values, name, float(tv)) for name, tv in taus.items()]
    return dist, tau_rows


def compute_group_metrics(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    taus: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"--group-col {group_col!r} not in dataframe.")
    gdf = df[[group_col, value_col]].dropna(subset=[value_col])
    gdf[group_col] = gdf[group_col].astype(str)

    dist_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []

    for g, sub in gdf.groupby(group_col, sort=False):
        vals = sub[value_col].to_numpy(dtype=np.float64)
        d = distribution_summary(vals)
        d[group_col] = g
        dist_rows.append(d)
        for name, tv in taus.items():
            row = tau_metrics(vals, name, float(tv))
            row[group_col] = g
            tau_rows.append(row)

    dist_df = pd.DataFrame(dist_rows)
    cols = [group_col] + [c for c in dist_df.columns if c != group_col]
    dist_df = dist_df[cols]
    tau_df = pd.DataFrame(tau_rows)
    cols_t = [group_col] + [c for c in tau_df.columns if c != group_col]
    tau_df = tau_df[cols_t]
    return dist_df, tau_df


def _rank_groups(
    group_tau: pd.DataFrame,
    group_col: str,
    metric: str,
    tau_name: str = "tau_league_mean",
    *,
    top: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = group_tau[group_tau["tau_name"] == tau_name].copy()
    if m.empty or metric not in m.columns:
        return pd.DataFrame(), pd.DataFrame()
    m = m.sort_values(metric, ascending=False, na_position="last")
    hi = m.head(top)[[group_col, metric]].reset_index(drop=True)
    lo = m.sort_values(metric, ascending=True, na_position="last").head(top)[[group_col, metric]].reset_index(
        drop=True
    )
    return hi, lo


def _dataframe_to_pipe_markdown(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    d = df if max_rows is None else df.iloc[: max_rows]
    cols = list(d.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{float(val):.8g}" if np.isfinite(val) else str(val))
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    out = "\n".join([head, sep] + rows)
    if max_rows is not None and len(df) > max_rows:
        out += f"\n\n_(showing first {max_rows} of {len(df)} rows)_"
    return out


def _dict_row_to_markdown_table(row: dict[str, Any]) -> str:
    lines = ["| metric | value |", "|---|---:|"]
    for k, v in row.items():
        if isinstance(v, float):
            cell = f"{v:.8g}" if np.isfinite(v) else str(v)
        else:
            cell = str(v)
        lines.append(f"| {k} | {cell} |")
    return "\n".join(lines)


def write_markdown_summary(
    *,
    input_description: str,
    value_col: str,
    n_rows: int,
    taus: dict[str, float],
    pooled_dist: dict[str, Any],
    pooled_tau: pd.DataFrame,
    group_col: str | None,
    group_dist: pd.DataFrame | None,
    group_tau: pd.DataFrame | None,
    out_path: Path,
) -> None:
    lines = [
        "# Value distribution robustness metrics",
        "",
        "## 1. Input",
        "",
        f"- **Input:** {input_description}",
        f"- **Value column:** `{value_col}`",
        f"- **Rows used (finite V):** {n_rows:,}",
        "",
        "## 2. Thresholds (league BIP, 2020–2026 Statcast run)",
        "",
        "| name | tau |",
        "|---|---:|",
    ]
    for k, v in taus.items():
        lines.append(f"| {k} | {v:.6f} |")
    lines.extend(["", "## 3. Pooled distribution summary", "", _dict_row_to_markdown_table(pooled_dist), ""])

    lines.extend(["", "## 4. Pooled tau-level metrics", ""])
    lines.append(_dataframe_to_pipe_markdown(pooled_tau))

    if group_col and group_dist is not None and group_tau is not None:
        lines.extend(
            [
                "",
                f"## 5. By-group summaries (`{group_col}`)",
                "",
                "### Distribution (one row per group)",
                "",
                _dataframe_to_pipe_markdown(group_dist),
                "",
                "### Tau metrics (long: group × tau)",
                "",
                _dataframe_to_pipe_markdown(group_tau, max_rows=500),
                "",
                "### Top / bottom groups (at `tau_league_mean`)",
                "",
            ]
        )
        for label, metric in [
            ("mean value (from distribution table)", "mean_value"),
            ("Omega up/down", "omega_up_down"),
            ("Sortino downside ratio", "sortino_downside_ratio"),
            ("Downside deviation", "downside_deviation"),
        ]:
            if metric == "mean_value":
                sub_hi = group_dist.sort_values("mean_value", ascending=False, na_position="last")
                hi = sub_hi.head(5)[[group_col, "mean_value"]]
                sub_lo = group_dist.sort_values("mean_value", ascending=True, na_position="last")
                lo = sub_lo.head(5)[[group_col, "mean_value"]]
            else:
                hi, lo = _rank_groups(group_tau, group_col, metric, "tau_league_mean", top=5)
            lines.append(f"#### {label}")
            lines.append("")
            lines.append("**Top:**")
            lines.append(_dataframe_to_pipe_markdown(hi))
            lines.append("")
            lines.append("**Bottom:**")
            lines.append(_dataframe_to_pipe_markdown(lo))
            lines.append("")

    lines.extend(
        [
            "## 6. Interpretation (plain language)",
            "",
            "- **mean_value:** central tendency of contact value `V` over the sample (overall quality level).",
            "- **std_value:** total dispersion of `V` (volatility including upside and downside).",
            "- **downside_deviation:** RMS of shortfalls `(tau - V)_+` — bad-side volatility relative to the threshold.",
            "- **sortino_downside_ratio:** `(mean(V) - tau) / downside_deviation` — excess mean value per unit "
            "downside RMS (higher is better if downside deviation is well-defined).",
            "- **omega_up_down:** `E[(V-tau)_+] / E[(tau-V)_+]` — expected upside above tau vs expected shortfall below tau.",
            "- **omega_down_up:** inverse ratio — loss-to-gain emphasis from the threshold.",
            "",
            "These are **distribution-level** summaries after `(EV, LA, SA)` have been integrated into `V` "
            "(row-level `V` already given).",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-path", type=Path, default=None, help="Single table (.parquet / .csv / .feather).")
    ap.add_argument(
        "--stack-run-root",
        type=Path,
        default=None,
        help="Directory of per-hitter subfolders each containing the same leaf file (default parquet name below).",
    )
    ap.add_argument(
        "--stack-filename",
        type=str,
        default="simulated_ev_la_sa_value.parquet",
        help="Leaf filename used with --stack-run-root.",
    )
    ap.add_argument("--value-col", type=str, default=None)
    ap.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Column for by-group metrics. With --stack-run-root, defaults to ``hitter_id`` (subdir slug).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "outputs" / "value_distribution_robustness_metrics",
    )
    args = ap.parse_args()

    if (args.input_path is None) == (args.stack_run_root is None):
        ap.error("Provide exactly one of --input-path or --stack-run-root.")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stack_run_root is not None:
        group_default = "hitter_id"
        group_col = args.group_col or group_default
        df, n_files = stack_value_tables_from_run_root(
            args.stack_run_root,
            filename=args.stack_filename,
            group_col_name=group_col,
        )
        root = args.stack_run_root.resolve()
        input_description = f"stacked **{n_files}** files `{args.stack_filename}` under `{root}` (group column `{group_col}`)"
    else:
        inp = args.input_path.resolve()
        input_description = f"`{inp}`"
        df = load_table(inp)
        group_col = args.group_col

    vcol = infer_value_col(df, args.value_col)
    v = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=np.float64)
    v_fin = v[np.isfinite(v)]
    n = int(len(v_fin))

    pooled_dist, pooled_tau_list = compute_pooled_metrics(v_fin, TAU_DEFAULTS)
    pooled_tau_df = pd.DataFrame(pooled_tau_list)

    pd.DataFrame([pooled_dist]).to_csv(out_dir / "pooled_distribution_summary.csv", index=False)
    pooled_tau_df.to_csv(out_dir / "pooled_tau_metrics.csv", index=False)

    group_dist_df = None
    group_tau_df = None
    if group_col:
        gd, gt = compute_group_metrics(df, vcol, group_col, TAU_DEFAULTS)
        group_dist_df = gd
        group_tau_df = gt
        gd.to_csv(out_dir / "group_distribution_summary.csv", index=False)
        gt.to_csv(out_dir / "group_tau_metrics.csv", index=False)

    md_path = out_dir / "robustness_metrics_summary.md"
    write_markdown_summary(
        input_description=input_description,
        value_col=vcol,
        n_rows=n,
        taus=TAU_DEFAULTS,
        pooled_dist=pooled_dist,
        pooled_tau=pooled_tau_df,
        group_col=group_col,
        group_dist=group_dist_df,
        group_tau=group_tau_df,
        out_path=md_path,
    )

    row_mean = pooled_tau_df.loc[pooled_tau_df["tau_name"] == "tau_league_mean"]
    omg = float(row_mean["omega_up_down"].iloc[0]) if len(row_mean) else float("nan")
    sor = float(row_mean["sortino_downside_ratio"].iloc[0]) if len(row_mean) else float("nan")

    print(out_dir)
    print(vcol)
    print(n)
    print(pooled_dist["mean_value"])
    print(pooled_dist["std_value"])
    print(omg)
    print(sor)
    print(md_path)


if __name__ == "__main__":
    main()
