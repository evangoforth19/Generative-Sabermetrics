#!/usr/bin/env python3
"""
Audit calibration run consistency across artifacts/diagnostics directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_fast(path: Path) -> str:
    size = path.stat().st_size
    if size <= 200 * 1024 * 1024:
        return sha256_file(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        head = f.read(2 * 1024 * 1024)
        if size > 4 * 1024 * 1024:
            f.seek(max(0, size - 2 * 1024 * 1024))
        tail = f.read(2 * 1024 * 1024)
    h.update(head)
    h.update(tail)
    h.update(str(size).encode("utf-8"))
    return f"fast:{h.hexdigest()}"


def read_tab(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def read_tab_cols(path: Path, usecols: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def available_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pq.ParquetFile(path).schema.names)
    return list(pd.read_csv(path, nrows=0, low_memory=False).columns)


def infer_cols(df: pd.DataFrame) -> dict[str, str | None]:
    eid = "event_id" if "event_id" in df.columns else None
    if eid is None and all(c in df.columns for c in ("game_pk", "at_bat_number", "pitch_number")):
        eid = "__event_id__"
        df[eid] = (
            df["game_pk"].astype(str)
            + "_"
            + df["at_bat_number"].astype(str)
            + "_"
            + df["pitch_number"].astype(str)
        )
    ev_dec = next((c for c in ["EV_dec", "ev_dec", "EV", "launch_speed_dec", "decoded_EV"] if c in df.columns), None)
    ev_cal = next((c for c in ["EV_cal", "ev_cal", "calibrated_EV"] if c in df.columns), None)
    ev_obs = next((c for c in ["EV_obs", "ev_obs", "observed_EV", "launch_speed"] if c in df.columns), None)
    return {"event_id": eid, "ev_dec": ev_dec, "ev_cal": ev_cal, "ev_obs": ev_obs}


def event_bias(df: pd.DataFrame, ev_col: str, obs_col: str, event_col: str) -> float:
    d = df.copy()
    d[event_col] = d[event_col].astype(str)
    d[ev_col] = pd.to_numeric(d[ev_col], errors="coerce")
    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    g = d.groupby(event_col, sort=False)
    mu = g[ev_col].mean()
    yo = g[obs_col].first()
    m = pd.concat([mu.rename("mu"), yo.rename("y")], axis=1).dropna()
    if m.empty:
        return float("nan")
    return float(np.mean(m["mu"] - m["y"]))


def rmse_mae(df: pd.DataFrame, pred_col: str, obs_col: str, event_col: str) -> tuple[float, float]:
    d = df.copy()
    d[event_col] = d[event_col].astype(str)
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    g = d.groupby(event_col, sort=False)
    mu = g[pred_col].mean()
    yo = g[obs_col].first()
    m = pd.concat([mu.rename("mu"), yo.rename("y")], axis=1).dropna()
    if m.empty:
        return float("nan"), float("nan")
    e = (m["mu"] - m["y"]).to_numpy(dtype=float)
    return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e)))


def sample_crps(s: np.ndarray, y: float) -> float:
    s = s[np.isfinite(s)]
    if s.size < 2 or not np.isfinite(y):
        return float("nan")
    e1 = np.mean(np.abs(s - y))
    xs = np.sort(s.astype(float))
    n = xs.size
    i = np.arange(1, n + 1, dtype=float)
    # Exact mean pairwise absolute difference in O(n log n), no NxN matrix.
    e2 = (2.0 / (n * n)) * np.sum((2.0 * i - n - 1.0) * xs)
    return float(e1 - 0.5 * e2)


def interval_cov(df: pd.DataFrame, ev_col: str, obs_col: str, event_col: str, alpha: float = 0.90) -> float:
    d = df.copy()
    d[event_col] = d[event_col].astype(str)
    d[ev_col] = pd.to_numeric(d[ev_col], errors="coerce")
    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    ql = (1.0 - alpha) / 2.0
    qh = 1.0 - ql
    vals = []
    for _, g in d.groupby(event_col, sort=False):
        x = g[ev_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            continue
        y = float(pd.to_numeric(g[obs_col], errors="coerce").dropna().iloc[0]) if g[obs_col].notna().any() else float("nan")
        lo, hi = np.quantile(x, [ql, qh])
        vals.append(float(lo <= y <= hi))
    return float(np.mean(vals)) if vals else float("nan")


def crps_mean(df: pd.DataFrame, ev_col: str, obs_col: str, event_col: str, max_events: int = 5000, max_draws: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    d = df.copy()
    d[event_col] = d[event_col].astype(str)
    d[ev_col] = pd.to_numeric(d[ev_col], errors="coerce")
    d[obs_col] = pd.to_numeric(d[obs_col], errors="coerce")
    vals = []
    for i, (_, g) in enumerate(d.groupby(event_col, sort=False)):
        if i >= max_events:
            break
        x = g[ev_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size > max_draws:
            idx = rng.choice(x.size, size=max_draws, replace=False)
            x = x[idx]
        y = float(pd.to_numeric(g[obs_col], errors="coerce").dropna().iloc[0]) if g[obs_col].notna().any() else float("nan")
        vals.append(sample_crps(x, y))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def gather_file_meta(dir_path: Path) -> list[dict[str, Any]]:
    out = []
    if not dir_path.exists():
        return out
    wanted = {".parquet", ".csv", ".joblib", ".json"}
    for p in sorted([x for x in dir_path.rglob("*") if x.is_file() and x.suffix.lower() in wanted]):
        rec: dict[str, Any] = {
            "path": str(p),
            "size": int(p.stat().st_size),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            "sha256": sha256_fast(p),
        }
        if p.suffix.lower() in [".parquet", ".csv"]:
            try:
                # Avoid full reads for huge files.
                if p.suffix.lower() == ".parquet":
                    pf = pq.ParquetFile(p)
                    rec["shape"] = (pf.metadata.num_rows, pf.metadata.num_columns)
                    rec["columns"] = pf.schema.names
                    if p.stat().st_size <= 5 * 1024 * 1024:
                        dfh = pd.read_parquet(p).head(5)
                        rec["head5"] = dfh.to_dict(orient="records")
                    else:
                        rec["head5"] = "<skipped large parquet>"
                else:
                    if p.stat().st_size <= 5 * 1024 * 1024:
                        df = pd.read_csv(p, low_memory=False)
                        rec["shape"] = tuple(df.shape)
                        rec["columns"] = list(df.columns)
                        rec["head5"] = df.head(5).to_dict(orient="records")
                    else:
                        dfh = pd.read_csv(p, nrows=5, low_memory=False)
                        rec["shape"] = ("<large_csv_rows_unknown>", len(dfh.columns))
                        rec["columns"] = list(dfh.columns)
                        rec["head5"] = dfh.head(5).to_dict(orient="records")
            except Exception as e:
                rec["read_error"] = str(e)
        out.append(rec)
    return out


def print_file_meta(lines: list[str], title: str, metas: list[dict[str, Any]]) -> None:
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    if not metas:
        lines.append("No files found.")
        return
    for m in metas:
        lines.append(f"path: {m['path']}")
        lines.append(f"  size: {m['size']}")
        lines.append(f"  mtime: {m['mtime']}")
        lines.append(f"  sha256: {m['sha256']}")
        if "shape" in m:
            lines.append(f"  shape: {m['shape']}")
            lines.append(f"  cols: {m.get('columns', [])[:30]}")
            lines.append(f"  head5: {json.dumps(m.get('head5', []), default=str)[:500]}")
        if "read_error" in m:
            lines.append(f"  read_error: {m['read_error']}")


def find_calibrated_candidates(root: Path) -> list[Path]:
    out = []
    if not root.exists():
        return out
    names = ["cal_with_obs.parquet", "calibrated_draws_test.parquet", "calibrated_draws_validation.parquet"]
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in names:
            out.append(p)
            continue
        if p.suffix.lower() in [".parquet", ".csv"]:
            try:
                if p.suffix.lower() == ".parquet":
                    cols = pq.ParquetFile(p).schema.names
                else:
                    cols = pd.read_csv(p, nrows=1, low_memory=False).columns
                if "EV_cal" in cols:
                    out.append(p)
            except Exception:
                pass
    # dedupe preserve order
    seen: set[str] = set()
    uniq = []
    for p in out:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def summarize_cal_file(path: Path) -> dict[str, Any]:
    cols = available_columns(path)
    tmp = pd.DataFrame(columns=cols)
    c = infer_cols(tmp)
    needed = [x for x in [c["event_id"], c["ev_cal"], c["ev_obs"]] if x]
    d = read_tab_cols(path, needed) if needed else pd.DataFrame()
    out: dict[str, Any] = {
        "path": str(path),
        "shape": tuple(d.shape) if not d.empty else ("unknown", len(cols)),
        "columns": cols,
        "unique_events": int(d[c["event_id"]].astype(str).nunique()) if c["event_id"] else None,
    }
    if c["ev_cal"] and c["ev_cal"] in d.columns:
        x = pd.to_numeric(d[c["ev_cal"]], errors="coerce")
        out["mean_EV_cal"] = float(x.mean(skipna=True))
        out["max_EV_cal"] = float(x.max(skipna=True))
        out["p99_EV_cal"] = float(x.quantile(0.99))
    if c["ev_obs"] and c["ev_obs"] in d.columns:
        y = pd.to_numeric(d[c["ev_obs"]], errors="coerce")
        out["mean_EV_obs"] = float(y.mean(skipna=True))
    if c["ev_cal"] and c["ev_obs"] and c["event_id"]:
        out["event_level_bias_mean"] = event_bias(d, c["ev_cal"], c["ev_obs"], c["event_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-diagnostics-dir", type=Path, required=True)
    ap.add_argument("--new-calibration-dir", type=Path, required=True)
    ap.add_argument("--new-diagnostics-dir", type=Path, required=True)
    ap.add_argument("--official-uncal-path", type=Path, required=True)
    ap.add_argument("--output-path", type=Path, required=True)
    args = ap.parse_args()

    lines: list[str] = []
    lines.append("RUN CONSISTENCY AUDIT")
    lines.append("=" * 80)
    lines.append(f"old_diagnostics_dir: {args.old_diagnostics_dir.resolve()}")
    lines.append(f"new_calibration_dir: {args.new_calibration_dir.resolve()}")
    lines.append(f"new_diagnostics_dir: {args.new_diagnostics_dir.resolve()}")
    lines.append(f"official_uncal_path: {args.official_uncal_path.resolve()}")
    lines.append(f"output_path: {args.output_path.resolve()}")

    old_cal_dir = args.new_calibration_dir.parent / "20260508_0319Z"
    dirs = [old_cal_dir, args.new_calibration_dir, args.old_diagnostics_dir, args.new_diagnostics_dir]

    lines.append("")
    lines.append("1) FILE EXISTENCE AND METADATA")
    lines.append("-" * 80)
    dir_metas: dict[str, list[dict[str, Any]]] = {}
    for d in dirs:
        metas = gather_file_meta(d)
        dir_metas[str(d.resolve())] = metas
        print_file_meta(lines, f"Directory: {d.resolve()}", metas)

    lines.append("")
    lines.append("2) LOCATE CALIBRATED DRAW FILES")
    lines.append("-" * 80)
    cands = []
    for d in dirs:
        cands.extend(find_calibrated_candidates(d))
    # dedupe
    seen = set()
    uniq = []
    for p in cands:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    cand_summ = []
    for p in uniq:
        try:
            s = summarize_cal_file(p)
            cand_summ.append(s)
            lines.append(json.dumps(s, default=str))
        except Exception as e:
            lines.append(f"failed to summarize {p}: {e}")

    lines.append("")
    lines.append("3) COMPARE OLD VS NEW CALIBRATED FILES")
    lines.append("-" * 80)
    old_cal_with_obs = args.old_diagnostics_dir / "cal_with_obs.parquet"
    new_cal_with_obs = args.new_diagnostics_dir / "cal_with_obs.parquet"
    if old_cal_with_obs.exists() and new_cal_with_obs.exists():
        ho = sha256_file(old_cal_with_obs)
        hn = sha256_file(new_cal_with_obs)
        lines.append(f"old_cal_with_obs hash: {ho}")
        lines.append(f"new_cal_with_obs hash: {hn}")
        lines.append(f"hash_equal: {ho == hn}")
        cols_o = available_columns(old_cal_with_obs)
        cols_n = available_columns(new_cal_with_obs)
        co0 = infer_cols(pd.DataFrame(columns=cols_o))
        cn0 = infer_cols(pd.DataFrame(columns=cols_n))
        need_o = [x for x in [co0["event_id"], co0["ev_cal"], co0["ev_obs"]] if x]
        need_n = [x for x in [cn0["event_id"], cn0["ev_cal"], cn0["ev_obs"]] if x]
        do = read_tab_cols(old_cal_with_obs, need_o)
        dn = read_tab_cols(new_cal_with_obs, need_n)
        lines.append(f"old shape: {do.shape}, new shape: {dn.shape}")
        co = infer_cols(do)
        cn = infer_cols(dn)
        eo = event_bias(do, co["ev_cal"], co["ev_obs"], co["event_id"]) if co["ev_cal"] and co["ev_obs"] and co["event_id"] else float("nan")
        en = event_bias(dn, cn["ev_cal"], cn["ev_obs"], cn["event_id"]) if cn["ev_cal"] and cn["ev_obs"] and cn["event_id"] else float("nan")
        lines.append(f"event-level mean bias old/new: {eo:.6f} / {en:.6f}")
        lines.append(f"mean EV_cal old/new: {pd.to_numeric(do[co['ev_cal']],errors='coerce').mean():.6f} / {pd.to_numeric(dn[cn['ev_cal']],errors='coerce').mean():.6f}")
        common_cols = [c for c in [co["event_id"], co["ev_cal"], co["ev_obs"]] if c and c in do.columns and c in dn.columns]
        if len(common_cols) >= 2:
            mo = do[[co["event_id"], co["ev_cal"]]].copy()
            mn = dn[[cn["event_id"], cn["ev_cal"]]].copy()
            mo.columns = ["event_id", "EV_cal_old"]
            mn.columns = ["event_id", "EV_cal_new"]
            # Align by per-event draw order to avoid cartesian explosion on merge.
            mo["draw_idx"] = mo.groupby("event_id").cumcount()
            mn["draw_idx"] = mn.groupby("event_id").cumcount()
            mj = mo.merge(mn, on=["event_id", "draw_idx"], how="inner")
            diff = pd.to_numeric(mj["EV_cal_new"], errors="coerce") - pd.to_numeric(mj["EV_cal_old"], errors="coerce")
            lines.append(f"rows where EV_cal differs: {int(np.sum(np.abs(diff.fillna(0)) > 1e-12))} / {len(mj)}")
            lines.append(
                "EV_cal_new - EV_cal_old summary: "
                f"mean={diff.mean():.6f}, std={diff.std():.6f}, p01={diff.quantile(0.01):.6f}, "
                f"p50={diff.quantile(0.50):.6f}, p99={diff.quantile(0.99):.6f}"
            )
    else:
        lines.append("old/new cal_with_obs.parquet not both present; skipping direct compare.")

    lines.append("")
    lines.append("4) RECOMPUTE OFFICIAL-TEST METRICS FROM EXACT NEW FILES")
    lines.append("-" * 80)
    if not args.official_uncal_path.exists():
        lines.append("official uncal path missing; cannot recompute.")
    else:
        # Prefer new diagnostics cal_with_obs if present, else new calibration calibrated_draws_test
        cal_path = new_cal_with_obs if new_cal_with_obs.exists() else (args.new_calibration_dir / "calibrated_draws_test.parquet")
        lines.append(f"using uncal path: {args.official_uncal_path.resolve()}")
        lines.append(f"using cal path: {cal_path.resolve()} exists={cal_path.exists()}")
        # Load only required columns.
        cols_u = available_columns(args.official_uncal_path)
        cols_c = available_columns(cal_path)
        cu0 = infer_cols(pd.DataFrame(columns=cols_u))
        cc0 = infer_cols(pd.DataFrame(columns=cols_c))
        need_u = [x for x in [cu0["event_id"], cu0["ev_dec"], cu0["ev_obs"]] if x]
        need_c = [x for x in [cc0["event_id"], cc0["ev_cal"], cc0["ev_obs"]] if x]
        du = read_tab_cols(args.official_uncal_path, need_u)
        dc = read_tab_cols(cal_path, need_c)
        cu = infer_cols(du)
        cc = infer_cols(dc)
        if cu["ev_obs"] is None and "observed_EV" in du.columns:
            cu["ev_obs"] = "observed_EV"
        if cc["ev_obs"] is None and "observed_EV" in dc.columns:
            cc["ev_obs"] = "observed_EV"
        # normalize event ids
        du[cu["event_id"]] = du[cu["event_id"]].astype(str)
        dc[cc["event_id"]] = dc[cc["event_id"]].astype(str)
        common = set(du[cu["event_id"]].unique()) & set(dc[cc["event_id"]].unique())
        du = du[du[cu["event_id"]].isin(common)].copy()
        dc = dc[dc[cc["event_id"]].isin(common)].copy()
        # event-level mean metrics
        eb = event_bias(du, cu["ev_dec"], cu["ev_obs"], cu["event_id"])
        ea = event_bias(dc, cc["ev_cal"], cc["ev_obs"], cc["event_id"])
        rmse_b, mae_b = rmse_mae(du, cu["ev_dec"], cu["ev_obs"], cu["event_id"])
        rmse_a, mae_a = rmse_mae(dc, cc["ev_cal"], cc["ev_obs"], cc["event_id"])
        crps_b = crps_mean(du, cu["ev_dec"], cu["ev_obs"], cu["event_id"])
        crps_a = crps_mean(dc, cc["ev_cal"], cc["ev_obs"], cc["event_id"])
        cov90_b = interval_cov(du, cu["ev_dec"], cu["ev_obs"], cu["event_id"], alpha=0.90)
        cov90_a = interval_cov(dc, cc["ev_cal"], cc["ev_obs"], cc["event_id"], alpha=0.90)
        xbu = pd.to_numeric(du[cu["ev_dec"]], errors="coerce").to_numpy(dtype=float)
        xca = pd.to_numeric(dc[cc["ev_cal"]], errors="coerce").to_numpy(dtype=float)
        p115_b = float(np.nanmean(xbu > 115))
        p115_a = float(np.nanmean(xca > 115))
        lines.append(f"event-level bias before/after: {eb:.6f} -> {ea:.6f}")
        lines.append(f"RMSE before/after: {rmse_b:.6f} -> {rmse_a:.6f}")
        lines.append(f"MAE before/after: {mae_b:.6f} -> {mae_a:.6f}")
        lines.append(f"CRPS before/after: {crps_b:.6f} -> {crps_a:.6f}")
        lines.append(f"90% coverage before/after: {cov90_b:.6f} -> {cov90_a:.6f}")
        lines.append(f"P(EV>115) before/after: {p115_b:.6f} -> {p115_a:.6f}")
        lines.append(f"max EV before/after: {np.nanmax(xbu):.6f} -> {np.nanmax(xca):.6f}")

    lines.append("")
    lines.append("5) EVALUATE INVOCATION ASSUMPTIONS")
    lines.append("-" * 80)
    for p in [args.new_diagnostics_dir / "metrics_summary.json", args.new_calibration_dir / "config_resolved.yaml", args.new_calibration_dir / "metrics_test.json", args.new_calibration_dir / "run_manifest.json"]:
        if p.exists():
            lines.append(f"found: {p}")
            if p.suffix.lower() == ".json":
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                    lines.append(json.dumps(j, indent=2)[:5000])
                except Exception as e:
                    lines.append(f"json parse failed: {e}")
            else:
                lines.append(p.read_text(encoding="utf-8")[:5000])
        else:
            lines.append(f"missing: {p}")

    lines.append("")
    lines.append("6) HARD WARNINGS")
    lines.append("-" * 80)
    warnings = []
    # identical file checks old/new diagnostics
    old_files = {Path(m["path"]).name: m for m in dir_metas.get(str(args.old_diagnostics_dir.resolve()), [])}
    new_files = {Path(m["path"]).name: m for m in dir_metas.get(str(args.new_diagnostics_dir.resolve()), [])}
    common_names = sorted(set(old_files) & set(new_files))
    identical_names = [n for n in common_names if old_files[n]["sha256"] == new_files[n]["sha256"]]
    if identical_names:
        warnings.append(f"new diagnostics contains files identical to old diagnostics: {identical_names[:50]}")
    # stale / path checks
    if new_cal_with_obs.exists() and old_cal_with_obs.exists() and sha256_file(new_cal_with_obs) == sha256_file(old_cal_with_obs):
        warnings.append("new cal_with_obs.parquet is identical to old cal_with_obs.parquet")
    # mean bias contradiction check
    try:
        if new_cal_with_obs.exists():
            cols_n = available_columns(new_cal_with_obs)
            cn0 = infer_cols(pd.DataFrame(columns=cols_n))
            need_n = [x for x in [cn0["event_id"], cn0["ev_cal"], cn0["ev_obs"]] if x]
            dn = read_tab_cols(new_cal_with_obs, need_n)
            cn = infer_cols(dn)
            bias_file = event_bias(dn, cn["ev_cal"], cn["ev_obs"], cn["event_id"]) if cn["ev_cal"] and cn["ev_obs"] and cn["event_id"] else float("nan")
            ms = args.new_diagnostics_dir / "metrics_summary.json"
            if ms.exists():
                j = json.loads(ms.read_text(encoding="utf-8"))
                bjson = j.get("mean_metrics", {}).get("bias_after", None)
                if bjson is not None and np.isfinite(float(bjson)) and np.isfinite(float(bias_file)):
                    if abs(float(bjson) - float(bias_file)) > 1e-3:
                        warnings.append(
                            f"mean after-bias contradiction: file-derived={bias_file:.6f} vs metrics_summary.json={float(bjson):.6f}"
                        )
    except Exception as e:
        warnings.append(f"bias contradiction check failed: {e}")

    if not warnings:
        lines.append("No hard warnings triggered.")
    else:
        for w in warnings:
            lines.append(f"WARNING: {w}")

    lines.append("")
    lines.append("7) FINAL DIAGNOSIS")
    lines.append("-" * 80)
    diagnosis = "F. something else"
    diag_reason = []
    if warnings:
        if any("identical" in w.lower() for w in warnings):
            diagnosis = "A. stale diagnostics"
            diag_reason.append("new diagnostics files/hash overlap old diagnostics")
        if any("contradiction" in w.lower() for w in warnings):
            diagnosis = "E. metric computation bug"
            diag_reason.append("metrics_summary bias conflicts with direct file recomputation")
    else:
        diag_reason.append("new files and recomputed metrics are internally consistent")
    lines.append(f"Diagnosis category: {diagnosis}")
    lines.append("Reasoning: " + "; ".join(diag_reason))

    outp = args.output_path.resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(outp))


if __name__ == "__main__":
    main()

