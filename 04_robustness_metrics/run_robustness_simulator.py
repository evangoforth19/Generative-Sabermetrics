#!/usr/bin/env python3
"""
Robustness simulator: for each hitter and each of 57 pitch-context GMM components,
adaptively sample (EV, LA, SA) via frozen stage-u / stage-z + physics decoder,
then apply the batted-ball value surface (xwOBAcon_3D) when available.

Outputs under this directory (folder name: Robustness Simulator):
  artifacts/          — cached value predictor, cluster manifests
  results/parquet/    — one file per hitter with all cluster draws
  results/summary/    — CSV summaries (mean/var by cluster)

Default cluster counts match user_fixed_k_map.json (total 57 components).
Uses decoder admissibility="lenient" for forward generative benchmarking (still drops
true kinematic impossibilities). See physics_decoder.decode_bip_from_sample docstring.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import torch

# -----------------------------------------------------------------------------
# Repo layout
# -----------------------------------------------------------------------------

_here = Path(__file__).resolve()
REPO = _here.parents[1]
MMC2 = REPO / "MCMC 2"
SBI = MMC2 / "sbi_forward_sim"
OUT_ROOT = _here.parent

if str(MMC2) not in sys.path:
    sys.path.insert(0, str(MMC2))

from sbi_forward_sim.src.data_u import (  # noqa: E402
    load_all_u_data,
    load_standardization_stats,
    merge_player_constants,
)
from sbi_forward_sim.src.data_z import load_all_z_data_native_trunc_ex  # noqa: E402
from sbi_forward_sim.src.feature_contract_z import load_feature_contract_z  # noqa: E402
from sbi_forward_sim.src.heldout_forward_inference import (  # noqa: E402
    u_encoder_inputs_from_row,
    z_base_x_num_from_row,
    z_categoricals_from_row,
)
from sbi_forward_sim.src.models_u import SharedMixtureGaussianVonMisesUNet  # noqa: E402
from sbi_forward_sim.src.models_z import ConditionalHybridGaussVonMisesTruncExZ  # noqa: E402
from sbi_forward_sim.src.physics_decoder import decode_bip_batch  # noqa: E402
from sbi_forward_sim.src.physics_decoder_contract import DecoderInputs  # noqa: E402
from sbi_forward_sim.src.pipeline_heldout_sample import (  # noqa: E402
    patch_z_x_num_with_u,
    sample_u_shared_gaussian_vm,
    sample_z_hybrid_trunc_ex_batched,
)
from sbi_forward_sim.src.physics_calibration import CONTEXT_COLUMNS, EVPhysicsCalibrator  # noqa: E402

logger = logging.getLogger("run_robustness_simulator")

# Pitch groups (duplicate of scripts/pitch_context_gmm_pitch_groups.py)

GROUP_MAP: dict[str, list[str]] = {
    "4F": ["FF", "FA"],
    "2F": ["SI", "FT"],
    "CF": ["FC"],
    "S": ["SL", "ST"],
    "C": ["CU", "KC", "CS"],
    "CH": ["CH", "FS"],
}
GROUP_ORDER = ("4F", "2F", "CF", "S", "C", "CH")


def _pitch_type_to_group() -> dict[str, str]:
    o: dict[str, str] = {}
    for g, pts in GROUP_MAP.items():
        for pt in pts:
            o[pt.upper()] = g
    return o


PT_TO_GROUP = _pitch_type_to_group()
NUM_FEATURES = [
    "z_count",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "spin_axis_sin",
    "spin_axis_cos",
    "release_speed",
]


@dataclass(frozen=True)
class ClusterSpec:
    global_id: int
    pitch_group: str
    component: int


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_u_model(cfg: dict, vocabs: dict, device: torch.device) -> SharedMixtureGaussianVonMisesUNet:
    mcfg = cfg["model"]
    num_f = len(cfg["numeric_features"]) + len(cfg["player_constant_features"])
    vs = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return SharedMixtureGaussianVonMisesUNet(
        input_dim=num_f,
        vocab_sizes=vs,
        embedding_dims=emb,
        n_mixture=int(mcfg["n_mixture_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        kappa_max=float(mcfg.get("kappa_max", 120.0)),
    ).to(device)


def _build_z_trunc(cfg: dict, vocabs: dict, device: torch.device, z_fc: dict) -> ConditionalHybridGaussVonMisesTruncExZ:
    mcfg = cfg["model"]
    num_f = len(z_fc["x_numeric_zscore_column_order"])
    vs = {k: len(vocabs[k]) for k in sorted(vocabs.keys())}
    emb = {k: cfg["categorical_features"][k]["embedding_dim"] for k in sorted(vocabs.keys())}
    return ConditionalHybridGaussVonMisesTruncExZ(
        input_dim=num_f,
        vocab_sizes=vs,
        embedding_dims=emb,
        n_components=int(mcfg["n_components"]),
        hidden_width=int(mcfg["hidden_width"]),
        n_hidden=int(mcfg["hidden_layers"]),
        activation=mcfg.get("activation", "tanh"),
        dropout=float(cfg["training"].get("dropout", 0.0)),
        chol_eps=float(mcfg.get("chol_eps", 1e-5)),
        kappa_floor=float(mcfg.get("kappa_floor", 1e-3)),
        sigma_floor=float(mcfg.get("sigma_floor", 1e-3)),
    ).to(device)


def _prepare_stats_u(project_root: Path, u_cfg: dict[str, Any], master: pd.DataFrame) -> dict[str, dict[str, float]]:
    import copy

    stats = copy.deepcopy(load_standardization_stats(project_root / u_cfg["paths"]["standardization_stats"]))
    ut = pd.read_parquet(project_root / u_cfg["paths"]["u_train"])
    ut = merge_player_constants(ut, master)
    for c in u_cfg["player_constant_features"]:
        if c not in stats:
            v = pd.to_numeric(ut[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
    return stats


def _prepare_stats_z(
    project_root: Path,
    z_cfg: dict[str, Any],
    z_fc: dict[str, Any],
    master: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    import copy

    stats = copy.deepcopy(load_standardization_stats(project_root / z_cfg["paths"]["standardization_stats"]))
    zt = pd.read_parquet(project_root / z_cfg["paths"]["z_train"])
    zt = merge_player_constants(zt, master)
    for t in z_fc["targets"]:
        if t not in stats:
            v = pd.to_numeric(zt[t], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[t] = {"mean": float(v.mean(skipna=True)), "std": sig}
    for c in z_fc["player_constant_features"]:
        if c not in stats:
            v = pd.to_numeric(zt[c], errors="coerce")
            sig = float(v.std(skipna=True))
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1.0
            stats[c] = {"mean": float(v.mean(skipna=True)), "std": sig}
    return stats


def _g_dict(row: pd.Series) -> dict[str, Any]:
    from sbi_forward_sim.src.physics_decoder_contract import DECODER_G_NUMERIC_REQUIRED, DECODER_G_OPTIONAL, validate_decoder_context_g

    g: dict[str, Any] = {}
    for k in DECODER_G_NUMERIC_REQUIRED:
        g[k] = float(pd.to_numeric(row[k], errors="coerce"))
    if "spin_axis_deg" in row.index and pd.notna(row.get("spin_axis_deg")):
        g["spin_axis_deg"] = float(row["spin_axis_deg"])
    else:
        g["spin_axis_sin"] = float(row["spin_axis_sin"])
        g["spin_axis_cos"] = float(row["spin_axis_cos"])
    for ok in DECODER_G_OPTIONAL:
        if ok in row.index and pd.notna(row.get(ok)):
            g[ok] = float(row[ok])
    validate_decoder_context_g(g)
    return g


def _p_dict(row: pd.Series, hitter_meta: dict[str, dict[str, float]]) -> dict[str, float]:
    from sbi_forward_sim.src.physics_decoder_contract import DECODER_P_REQUIRED

    hn = str(row["batter_name"]).strip().lower()
    hm = hitter_meta.get(hn)
    if hm is None:
        raise KeyError(f"No hitter_meta row for batter_name={hn!r}")
    p: dict[str, float] = {}
    for k in DECODER_P_REQUIRED:
        p[k] = float(hm[k])
    return p


def _spray_trig_from_hc(hc_x: float, hc_y: float) -> float:
    HOME_X = 125.42
    HOME_Y = 198.27
    x = hc_x - HOME_X
    y = HOME_Y - hc_y
    return float(np.degrees(np.arctan2(x, y)))


def _train_value_surrogate(
    statcast_path: Path,
    out_clf_path: Path,
) -> tuple[Any, str]:
    """Train sklearn HGB on BIP outcomes when LightGBM bundle is absent."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = pd.read_parquet(statcast_path, columns=["launch_speed", "launch_angle", "hc_x", "hc_y", "events"])
    evt_map = {
        "field_out": "out",
        "force_out": "out",
        "grounded_into_double_play": "out",
        "double_play": "out",
        "fielders_choice_out": "out",
        "sac_fly": "out",
        "field_error": "out",
        "single": "single",
        "double": "double",
        "triple": "triple",
        "home_run": "home_run",
    }
    classes = ["out", "single", "double", "triple", "home_run"]
    w = np.array([0.0, 0.902, 1.279, 1.618, 2.078], dtype=np.float64)
    ds = df.dropna(subset=["launch_speed", "launch_angle", "hc_x", "hc_y", "events"]).copy()
    ds["cls"] = ds["events"].map(evt_map)
    ds = ds[ds["cls"].notna()].copy()
    sa = [_spray_trig_from_hc(float(r.hc_x), float(r.hc_y)) for r in ds.itertuples(index=False)]
    ds["sa"] = sa
    ds["y_code"] = ds["cls"].map({nm: i for i, nm in enumerate(classes)})
    ds = ds[ds["y_code"].notna()].reset_index(drop=True)
    rad = np.radians(ds["sa"].to_numpy())
    X = np.column_stack(
        [
            ds["launch_speed"].to_numpy(),
            ds["launch_angle"].to_numpy(),
            np.sin(rad),
            np.cos(rad),
        ]
    ).astype(np.float64)
    y = ds["y_code"].to_numpy(dtype=np.int64)
    clf = HistGradientBoostingClassifier(
        max_depth=12,
        max_iter=250,
        learning_rate=0.06,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    out_clf_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "xwoba_weights": w, "class_order": classes}, out_clf_path)
    return joblib.load(out_clf_path), "HistGradientBoosting_surrogate"


def _predict_xwoba(ev: np.ndarray, la: np.ndarray, sa: np.ndarray, bundle: dict[str, Any], mode: str) -> np.ndarray:
    if mode == "lightgbm":
        bv = OUT_ROOT.parent / "outputs" / "batted_ball_value_surface"
        sys.path.insert(0, str(bv))
        from batted_ball_value_model import predict_xwobacon3d  # type: ignore

        return np.asarray(predict_xwobacon3d(ev, la, sa, bundle, calibrated=True)).reshape(-1).astype(np.float64)

    clf = bundle["clf"]
    w = bundle["xwoba_weights"]
    rad = np.radians(sa.astype(np.float64))
    X = np.column_stack([ev.reshape(-1), la.reshape(-1), np.sin(rad), np.cos(rad)])
    P = clf.predict_proba(X)
    if P.shape[1] != len(w):
        raise ValueError(f"Classifier has {P.shape[1]} classes; expected {len(w)}")
    return (P @ w).reshape(-1).astype(np.float64)


def _stable_split(samples: np.ndarray, half: int, rtol: float) -> bool:
    if len(samples) < 2 * half:
        return False
    a = samples[-2 * half : -half]
    b = samples[-half:]
    s = float(np.std(np.r_[a, b], ddof=0))
    return bool(abs(float(np.mean(a) - np.mean(b))) <= rtol * max(s, 1e-9))


def _converged(ev: np.ndarray, la: np.ndarray, sa: np.ndarray, xw: np.ndarray, *, half: int, rtol: float) -> bool:
    return (
        _stable_split(ev, half, rtol)
        and _stable_split(la, half, rtol)
        and _stable_split(sa, half, rtol)
        and _stable_split(xw, half, rtol)
    )


def _sample_diag_component(
    gmm: Any, comp: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Scaled-space samples from single mixture component (full-cov approximation via diag cov)."""
    mean = np.asarray(gmm.means_[comp], dtype=np.float64)
    if gmm.covariance_type != "diag":
        # general case: use Cholesky of full cov
        cov = np.asarray(gmm.covariances_[comp], dtype=np.float64)
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
        z = rng.standard_normal(size=(n, mean.shape[0]))
        return z @ L.T + mean
    var = np.asarray(gmm.covariances_[comp], dtype=np.float64)
    z = rng.standard_normal(size=(n, mean.shape[0]))
    return mean + z * np.sqrt(np.clip(var, 1e-12, None))


def _save_hitter_distribution_png_robustness(df_long: pd.DataFrame, path: Path, *, title_slug: str) -> None:
    """2x2 density histograms for pooled cluster draws (EV, LA, SA, value)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df_long.shape[0] < 2:
        return
    ev = pd.to_numeric(df_long["EV"], errors="coerce").to_numpy(dtype=np.float64)
    la = pd.to_numeric(df_long["LA"], errors="coerce").to_numpy(dtype=np.float64)
    sa = pd.to_numeric(df_long["SA"], errors="coerce").to_numpy(dtype=np.float64)
    xv = pd.to_numeric(df_long["xwOBAcon"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ev) & np.isfinite(la) & np.isfinite(sa) & np.isfinite(xv)
    if int(ok.sum()) < 2:
        return
    ev, la, sa, xv = ev[ok], la[ok], sa[ok], xv[ok]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    bins = min(72, max(24, int(np.sqrt(float(ev.size)))))

    axes[0, 0].hist(ev, bins=bins, density=True, color="steelblue", alpha=0.85)
    axes[0, 0].set_title("EV (mph)")
    axes[0, 0].set_xlabel("EV")

    axes[0, 1].hist(la, bins=bins, density=True, color="darkorange", alpha=0.85)
    axes[0, 1].set_title("Launch angle (deg)")
    axes[0, 1].set_xlabel("LA")

    axes[1, 0].hist(sa, bins=bins, density=True, color="seagreen", alpha=0.85)
    axes[1, 0].set_title("Spray angle (deg)")
    axes[1, 0].set_xlabel("SA")

    axes[1, 1].hist(xv, bins=bins, density=True, color="purple", alpha=0.85)
    axes[1, 1].set_title("xwOBAcon")
    axes[1, 1].set_xlabel("xwOBAcon")

    fig.suptitle(title_slug.replace("_", " ").title(), fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _assign_pitch_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pt = out["pitch_type"].astype(str).str.strip().str.upper()
    out["pitch_group"] = pt.map(PT_TO_GROUP)
    return out


def _draws_df_for_calibration(
    ev_dec: np.ndarray,
    la: np.ndarray,
    sa: np.ndarray,
    *,
    calibrator: EVPhysicsCalibrator,
    cs: ClusterSpec,
    hitter: str,
    tmpl: pd.Series,
    pitch_type: str,
) -> pd.DataFrame:
    """One synthetic event_id per (hitter, cluster); context from GMM template row."""
    n = int(ev_dec.shape[0])
    eid = int(cs.global_id * 1_000_003 + (abs(hash((hitter, cs.global_id))) % 999_983))
    evc = calibrator.ev_col
    df = pd.DataFrame(
        {
            calibrator.group_col: np.full(n, eid, dtype=np.int64),
            evc: ev_dec.astype(np.float64),
            "LA": la.astype(np.float64),
            "SA": sa.astype(np.float64),
            "pitch_group": cs.pitch_group,
            "pitch_type": str(pitch_type),
            "draw_idx": np.arange(n, dtype=np.int32),
        }
    )
    for c in CONTEXT_COLUMNS:
        if c in ("pitch_group", "pitch_type"):
            continue
        if c in tmpl.index and c not in df.columns:
            df[c] = tmpl.get(c)
    if "spin_rate" not in df.columns and "release_spin_rate" in df.columns:
        df["spin_rate"] = pd.to_numeric(df["release_spin_rate"], errors="coerce")
    if "pitcher_hand" not in df.columns and "p_throws" in tmpl.index:
        df["pitcher_hand"] = tmpl["p_throws"]
    if "batter_hand" not in df.columns and "stand" in tmpl.index:
        df["batter_hand"] = tmpl["stand"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=SBI / "data_processed" / "sbi_context_event_master.parquet")
    ap.add_argument("--u-run", type=Path, default=SBI / "outputs" / "p_u_given_g" / "20260407_200302Z")
    ap.add_argument("--z-run", type=Path, default=SBI / "outputs" / "p_z_given_u_g" / "20260501_055904Z")
    ap.add_argument(
        "--k-map-json",
        type=Path,
        default=OUT_ROOT.parent / "outputs" / "pitch_context_gmm_pitch_groups" / "user_fixed_k_map.json",
    )
    ap.add_argument(
        "--fitted-gmm-dir",
        type=Path,
        default=OUT_ROOT.parent / "outputs" / "pitch_context_gmm_pitch_groups" / "fitted_k_user",
    )
    ap.add_argument("--hitter-meta", type=Path, default=MMC2 / "Outputs" / "hitter_meta.csv")
    ap.add_argument(
        "--statcast-for-surrogate",
        type=Path,
        default=REPO / "data" / "statcast_pybaseball" / "twelve_hitters_bip_full_statcast.parquet",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-decode", type=int, default=256)
    ap.add_argument(
        "--max-outer-iters",
        type=int,
        default=8000,
        help="Safety cap on sampling rounds per (hitter × cluster)",
    )
    ap.add_argument("--min-admissible", type=int, default=1200, help="Min admissible draws before testing convergence")
    ap.add_argument("--max-admissible", type=int, default=8000, help="Hard cap admissible draws per cluster*hitter")
    ap.add_argument("--convergence-half", type=int, default=700, help="Compare last two consecutive blocks of this size")
    ap.add_argument("--convergence-rtol", type=float, default=0.03, help="Mean shift threshold in units of pooled std")
    ap.add_argument("--clusters-max", type=int, default=None, help="Debug: limit number of clusters (starting at 0)")
    ap.add_argument("--hitters-max", type=int, default=None, help="Debug: limit number of hitters")
    ap.add_argument(
        "--physics-calibrator-path",
        type=Path,
        default=None,
        help="Optional EVPhysicsCalibrator joblib; calibrates decoder EV before xwOBAcon surface",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for artifacts/ and results/ (default: this Robustness Simulator folder).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_root = Path(args.output_root).resolve() if args.output_root else OUT_ROOT

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    artifacts_dir = output_root / "artifacts"
    results_parquet_dir = output_root / "results" / "parquet"
    summary_dir = output_root / "results" / "summary"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_parquet_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    k_map: dict[str, int] = json.loads(Path(args.k_map_json).read_text(encoding="utf-8"))

    master = pd.read_parquet(Path(args.master))
    master = _assign_pitch_group(master)
    master = master[master["pitch_group"].notna()].copy()

    cluster_specs: list[ClusterSpec] = []
    cluster_records: list[dict[str, Any]] = []
    template_rows: dict[int, pd.Series] = {}
    cluster_pitch_types: dict[int, str] = {}
    gmm_payloads: dict[str, Any] = {}
    gid = 0
    for g in GROUP_ORDER:
        k_exp = int(k_map[g])
        path = Path(args.fitted_gmm_dir) / f"pitch_group_{g}_k{k_exp}.joblib"
        payload = joblib.load(path)
        gmm_payloads[g] = payload

        tmpl_idx: dict[int, int] = {}
        sub = master[master["pitch_group"] == g].copy()
        if sub.shape[0] < max(50, k_exp):
            raise RuntimeError(f"Group {g}: only {sub.shape[0]} master rows (< need variety)")
        for c in NUM_FEATURES:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=NUM_FEATURES).reset_index(drop=True)
        if sub.shape[0] < k_exp:
            raise RuntimeError(f"Group {g}: after NA drop only {len(sub)} rows (< K)")
        X = sub[NUM_FEATURES].to_numpy(dtype=np.float64)
        Xs = payload["scaler"].transform(X)
        labels = np.asarray(payload["gmm"].predict(Xs), dtype=int)
        for j in range(k_exp):
            mask = labels == j
            n_j = int(mask.sum())
            if n_j <= 0:
                pos_first = 0
                pt_mode = str(sub.iloc[0]["pitch_type"]).strip().upper()
                n_eff = 0
            elif n_j < 5:
                pos_first = int(np.flatnonzero(mask)[0])
                pt_mode = str(sub.iloc[pos_first]["pitch_type"]).strip().upper()
                n_eff = n_j
            else:
                pos_first = int(np.flatnonzero(mask)[0])
                pt_mode = (
                    sub.loc[mask, "pitch_type"].astype(str).str.strip().str.upper().mode().iloc[0]
                )
                n_eff = n_j

            cluster_specs.append(ClusterSpec(global_id=gid, pitch_group=g, component=j))
            template_rows[gid] = sub.iloc[pos_first]
            cluster_pitch_types[gid] = str(pt_mode)
            cluster_records.append(
                {
                    "cluster_global_id": gid,
                    "pitch_group": g,
                    "component": int(j),
                    "n_master_rows_assign": n_eff,
                }
            )
            gid += 1
    (artifacts_dir / "cluster_manifest.json").write_text(
        json.dumps({"clusters": cluster_records, "total_clusters": gid}, indent=2),
        encoding="utf-8",
    )

    if args.clusters_max is not None:
        cluster_specs = cluster_specs[: int(args.clusters_max)]

    u_ckpt = _load_ckpt(Path(args.u_run) / "checkpoint.pt")
    z_ckpt = _load_ckpt(Path(args.z_run) / "checkpoint.pt")
    assert str(z_ckpt.get("model_family", "")) == "hybrid_gauss_vonmises_trunc_ex"
    project_root = SBI.resolve()
    u_cfg = u_ckpt["config"]
    z_cfg = z_ckpt["config"]
    u_vocabs = u_ckpt["vocabs"]
    z_vocabs = z_ckpt["vocabs"]
    z_frozen_path = Path(args.z_run) / "feature_contract_frozen.json"
    z_fc = (
        json.loads(z_frozen_path.read_text(encoding="utf-8"))
        if z_frozen_path.is_file()
        else load_feature_contract_z(z_cfg, project_root)
    )
    z_num_order = list(z_fc["x_numeric_zscore_column_order"])

    master_full_for_stats = pd.read_parquet(Path(args.master))
    stats_u = _prepare_stats_u(project_root, u_cfg, master_full_for_stats)
    stats_z = _prepare_stats_z(project_root, z_cfg, z_fc, master_full_for_stats)

    u_model = _build_u_model(u_cfg, u_vocabs, device)
    u_model.load_state_dict(u_ckpt["model_state"])
    u_model.eval()

    z_model = _build_z_trunc(z_cfg, z_vocabs, device, z_fc)
    z_model.load_state_dict(z_ckpt["model_state"])
    z_model.eval()

    train_u, _, _, _ = load_all_u_data(u_cfg, project_root, vocabs_override=u_vocabs)
    train_z, _, _, _ = load_all_z_data_native_trunc_ex(
        z_cfg,
        project_root,
        vocabs_override=z_vocabs,
        feature_contract=z_fc,
    )

    T_va = float(u_ckpt["temperature_T_va"])
    T_ang_u = float(u_ckpt["temperature_T_ang"])
    kappa_max_u = float(u_ckpt.get("kappa_max", u_cfg["model"].get("kappa_max", 120.0)))
    T_gauss_z = float(z_ckpt["temperature_T_gauss"])
    T_ang_z = float(z_ckpt["temperature_T_ang"])
    tgxf = float(z_ckpt["temperature_T_gauss_x"]) if z_ckpt.get("temperature_T_gauss_x") is not None else None
    tgyf = float(z_ckpt["temperature_T_gauss_y"]) if z_ckpt.get("temperature_T_gauss_y") is not None else None

    tm_u = torch.tensor(train_u.target_means, dtype=torch.float32, device=device)
    ts_u = torch.tensor(train_u.target_stds, dtype=torch.float32, device=device)
    gm_z = torch.tensor(train_z.gauss_means, dtype=torch.float32, device=device)
    gs_z = torch.tensor(train_z.gauss_stds, dtype=torch.float32, device=device)

    g_gen_u = torch.Generator(device=device)
    g_gen_u.manual_seed(args.seed)
    g_gen_z = torch.Generator(device=device)
    g_gen_z.manual_seed(args.seed + 9101)

    u_cols_tpl = ("v_ss_tilde", "a_tilde", "d_tilde")

    # hitter meta keyed lower
    hm_df = pd.read_csv(Path(args.hitter_meta))
    hitter_meta = {}
    for r in hm_df.itertuples(index=False):
        hn = str(getattr(r, "batter_name")).strip().lower()
        hitter_meta[hn] = {
            "bat_length_in": float(getattr(r, "L_in")),
            "bat_weight_oz": float(getattr(r, "W_oz")),
            "x_cm_fixed_in": float(getattr(r, "x_cm_fixed_in")),
            "r_g_fixed_in": float(getattr(r, "r_g_fixed_in")),
            "Iz_oz_in2": float(getattr(r, "Iz_oz_in2")),
        }

    bv_dir = OUT_ROOT.parent / "outputs" / "batted_ball_value_surface"
    surrogate_path = artifacts_dir / "value_hgb_surrogate.pkl"
    if (bv_dir / "model_lgbm.pkl").is_file():
        sys.path.insert(0, str(bv_dir))
        from batted_ball_value_model import load_batted_ball_value_model  # type: ignore

        value_bundle = load_batted_ball_value_model(bv_dir)
        val_mode = "lightgbm"
    elif surrogate_path.is_file():
        value_bundle = joblib.load(surrogate_path)
        val_mode = "sklearn_hgb_surrogate"
    else:
        print("Training HistGradientBoosting value surrogate on Statcast BIP (once, cached)...")
        value_bundle, val_mode = _train_value_surrogate(
            Path(args.statcast_for_surrogate),
            surrogate_path,
        )

    val_meta = {"value_mode": val_mode, "value_bundle_path": str(bv_dir), "surrogate_path": str(surrogate_path)}
    (artifacts_dir / "value_model_meta.json").write_text(json.dumps(val_meta, indent=2), encoding="utf-8")

    physics_calibrator: EVPhysicsCalibrator | None = None
    if args.physics_calibrator_path is not None:
        cal_path = Path(args.physics_calibrator_path).resolve()
        if not cal_path.is_file():
            raise FileNotFoundError(f"--physics-calibrator-path not found: {cal_path}")
        physics_calibrator = EVPhysicsCalibrator.load(cal_path)
        logger.info("Loaded physics calibrator from %s", cal_path)

    # Stage-u vocab must recognize batter_name tokens
    vocab_batters = set(u_vocabs["batter_name"].keys()) - {"<UNK>"}
    hitters = sorted(h for h in hitter_meta if h in vocab_batters)
    if args.hitters_max is not None:
        hitters = hitters[: int(args.hitters_max)]

    def synth_row(
        template: pd.Series,
        feats7: np.ndarray,
        hitter: str,
        pitch_t: str,
        eid: int,
    ) -> pd.Series:
        r = template.copy()
        for i, cname in enumerate(NUM_FEATURES):
            r[cname] = float(feats7[i])
        r["pitch_type"] = pitch_t
        r["batter_name"] = hitter
        hm = hitter_meta[hitter]
        r["bat_length_in"] = hm["bat_length_in"]
        r["bat_weight_oz"] = hm["bat_weight_oz"]
        r["x_cm_fixed_in"] = hm["x_cm_fixed_in"]
        r["r_g_fixed_in"] = hm["r_g_fixed_in"]
        r["Iz_oz_in2"] = hm["Iz_oz_in2"]
        if "spin_axis_sin" in r.index and "spin_axis_cos" in r.index:
            sn, cs = float(r["spin_axis_sin"]), float(r["spin_axis_cos"])
            r["spin_axis_deg"] = float(np.degrees(np.arctan2(sn, cs)))
        r["event_id"] = int(eid)
        return r

    def run_cluster_hitter(
        hitter: str, cs: ClusterSpec
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        pay = gmm_payloads[cs.pitch_group]
        gmm, scaler_m = pay["gmm"], pay["scaler"]
        tmpl = template_rows[cs.global_id]
        pt_m = cluster_pitch_types[cs.global_id]

        ev_acc: list[float] = []
        la_acc: list[float] = []
        sa_acc: list[float] = []
        xw_acc: list[float] = []
        diag: dict[str, Any] = {
            "n_attempted": 0,
            "n_admissible": 0,
            "converged": False,
            "stopped_reason": "",
            "n_soft_capped": 0,
            "max_EV_before": float("nan"),
            "max_EV_after": float("nan"),
            "p99_EV_before": float("nan"),
            "p99_EV_after": float("nan"),
        }

        eid_base = -(cs.global_id * 500_001 + abs(hash(hitter)) % 100_000)
        outer = 0

        while len(ev_acc) < args.max_admissible:
            outer += 1
            if outer > int(args.max_outer_iters):
                diag["stopped_reason"] = "max_outer_iters"
                break

            b = max(1, int(args.batch_decode))
            Zh = _sample_diag_component(gmm, cs.component, b, rng)
            Xorig = scaler_m.inverse_transform(Zh)

            rows_dec: list[DecoderInputs] = []
            diag["n_attempted"] += b

            for ii in range(b):
                rr = synth_row(tmpl, Xorig[ii], hitter, pt_m, eid_base + outer * 10_000 + ii)

                try:
                    x_u_np, cat_u = u_encoder_inputs_from_row(rr, stats_u, u_vocabs, u_cfg)
                    x_u_t = torch.from_numpy(x_u_np).float().to(device)
                    cat_u_t = {
                        kk: torch.from_numpy(cat_u[kk]).long().to(device) for kk in sorted(cat_u.keys())
                    }

                    up = sample_u_shared_gaussian_vm(
                        u_model,
                        x_u_t,
                        cat_u_t,
                        n_samples=1,
                        T_va=T_va,
                        T_ang=T_ang_u,
                        kappa_max=kappa_max_u,
                        target_means=tm_u,
                        target_stds=ts_u,
                        generator=g_gen_u,
                    )
                    uraw = np.array(
                        [[float(up["v_ss_tilde"][0]), float(up["a_tilde"][0]), float(up["d_tilde"][0])]],
                        dtype=np.float32,
                    )
                    z_base = z_base_x_num_from_row(rr, z_num_order, stats_z, u_cols_tpl).astype(np.float32)
                    z_pat = patch_z_x_num_with_u(z_base[None, :], z_num_order, u_cols_tpl, uraw, stats_z)
                    cat_z_np = z_categoricals_from_row(rr, z_vocabs)

                    z_batch = torch.from_numpy(z_pat).float().to(device)
                    cat_z_t = {kk: torch.from_numpy(cat_z_np[kk]).long().to(device) for kk in sorted(cat_z_np.keys())}
                    zp = sample_z_hybrid_trunc_ex_batched(
                        z_model,
                        z_batch,
                        cat_z_t,
                        n_samples_per_row=1,
                        T_gauss=T_gauss_z,
                        T_gauss_x=tgxf,
                        T_gauss_y=tgyf,
                        T_ang=T_ang_z,
                        gauss_means=gm_z,
                        gauss_stds=gs_z,
                        generator=g_gen_z,
                    )
                    xd = zp["x"][0, 0].detach().cpu().item()
                    ey = zp["e_y_star"][0, 0].detach().cpu().item()
                    psi = zp["psi_deg"][0, 0].detach().cpu().item()
                    exv = zp["e_x"][0, 0].detach().cpu().item()

                    g_dec = _g_dict(rr)
                    p_dec = _p_dict(rr, hitter_meta)
                    dj = DecoderInputs(
                        event_id=int(rr["event_id"]),
                        batter_name=str(rr["batter_name"]),
                        g=g_dec,
                        p=p_dec,
                        u={
                            "v_ss_tilde": float(uraw[0, 0]),
                            "a_tilde": float(uraw[0, 1]),
                            "d_tilde": float(uraw[0, 2]),
                        },
                        z={"x": float(xd), "e_y_star": float(ey), "psi_deg": float(psi), "e_x": float(exv)},
                    )
                    rows_dec.append(dj)
                except Exception:
                    continue

            if not rows_dec:
                continue

            dec_df = decode_bip_batch(rows_dec, admissibility="lenient")
            adm = dec_df[dec_df["admissible"].eq(True)].copy()
            if adm.empty:
                continue
            evb = adm["EV"].to_numpy(dtype=np.float64)
            lab = adm["LA"].to_numpy(dtype=np.float64)
            sab = adm["SA"].to_numpy(dtype=np.float64)
            finite = np.isfinite(evb) & np.isfinite(lab) & np.isfinite(sab)
            evb, lab, sab = evb[finite], lab[finite], sab[finite]
            if evb.size == 0:
                continue
            xwb = _predict_xwoba(evb, lab, sab, value_bundle, val_mode)

            ev_acc.extend(evb.tolist())
            la_acc.extend(lab.tolist())
            sa_acc.extend(sab.tolist())
            xw_acc.extend(xwb.tolist())
            diag["n_admissible"] = len(ev_acc)

            if len(ev_acc) >= args.min_admissible:
                ev_a = np.asarray(ev_acc)
                la_a = np.asarray(la_acc)
                sa_a = np.asarray(sa_acc)
                xw_a = np.asarray(xw_acc)
                if _converged(ev_a, la_a, sa_a, xw_a, half=args.convergence_half, rtol=args.convergence_rtol):
                    diag["converged"] = True
                    diag["stopped_reason"] = "split_half_stability"
                    break

            if len(ev_acc) >= args.max_admissible:
                diag["stopped_reason"] = "max_admissible"
                break

        if not diag["stopped_reason"]:
            diag["stopped_reason"] = (
                "min_admissible_not_reached_timeout" if len(ev_acc) < args.min_admissible else "unknown"
            )

        ev_dec = np.asarray(ev_acc, dtype=np.float64)
        la_a = np.asarray(la_acc, dtype=np.float64)
        sa_a = np.asarray(sa_acc, dtype=np.float64)

        if physics_calibrator is not None and ev_dec.size:
            c_df = _draws_df_for_calibration(
                ev_dec,
                la_a,
                sa_a,
                calibrator=physics_calibrator,
                cs=cs,
                hitter=hitter,
                tmpl=tmpl,
                pitch_type=pt_m,
            )
            out_c = physics_calibrator.transform_draws(c_df, group_col=physics_calibrator.group_col)
            ev_cal = pd.to_numeric(out_c["EV_cal"], errors="coerce").to_numpy(dtype=np.float64)
            ev_pre = pd.to_numeric(out_c[physics_calibrator.ev_col], errors="coerce").to_numpy(dtype=np.float64)
            dm = out_c["EV_dec_mean_merged"].to_numpy(dtype=np.float64)
            cm = out_c["EV_cal_mean_merged"].to_numpy(dtype=np.float64)
            temp = cm + float(physics_calibrator.rho) * (ev_pre - dm)
            diag["n_soft_capped"] = int(np.sum(ev_cal < temp - 1e-4))
            diag["max_EV_before"] = float(np.nanmax(ev_dec))
            diag["max_EV_after"] = float(np.nanmax(ev_cal))
            diag["p99_EV_before"] = float(np.nanquantile(ev_dec, 0.99)) if ev_dec.size else float("nan")
            diag["p99_EV_after"] = float(np.nanquantile(ev_cal, 0.99)) if ev_cal.size else float("nan")
            xw_out = _predict_xwoba(ev_cal, la_a, sa_a, value_bundle, val_mode)
            logger.info(
                "Calibrator cluster %s hitter %s: max_EV %.2f -> %.2f p99 %.2f -> %.2f n_soft_capped=%s",
                cs.global_id,
                hitter,
                diag["max_EV_before"],
                diag["max_EV_after"],
                diag["p99_EV_before"],
                diag["p99_EV_after"],
                diag["n_soft_capped"],
            )
            return ev_dec, ev_cal, la_a, sa_a, xw_out, diag

        xw_out = np.asarray(xw_acc, dtype=np.float64)
        if ev_dec.size:
            diag["max_EV_before"] = float(np.nanmax(ev_dec))
            diag["max_EV_after"] = diag["max_EV_before"]
            diag["p99_EV_before"] = float(np.nanquantile(ev_dec, 0.99))
            diag["p99_EV_after"] = diag["p99_EV_before"]
        return ev_dec, ev_dec.copy(), la_a, sa_a, xw_out, diag

    manifest_run = {
        "n_hitters": len(hitters),
        "n_clusters": len(cluster_specs),
        "min_admissible": args.min_admissible,
        "max_admissible": args.max_admissible,
        "decoder": "lenient",
        "physics_calibrator_path": str(Path(args.physics_calibrator_path).resolve()) if args.physics_calibrator_path else None,
    }
    (artifacts_dir / "run_manifest.json").write_text(json.dumps(manifest_run, indent=2), encoding="utf-8")
    if physics_calibrator is None:
        logger.info("Physics calibrator: not used")
    else:
        logger.info("Physics calibrator active: %s", Path(args.physics_calibrator_path).resolve())

    for hi, hitter in enumerate(hitters):
        print(f"hitter {hi+1}/{len(hitters)}: {hitter}")
        rows_out: list[dict[str, Any]] = []
        sum_rows: list[dict[str, Any]] = []
        for cs in cluster_specs:
            ev_dec, ev_cal, la, sa, xw, dgi = run_cluster_hitter(hitter, cs)
            for k, (ed, ec, v2, v3, v4) in enumerate(zip(ev_dec, ev_cal, la, sa, xw)):
                row: dict[str, Any] = {
                    "batter_name": hitter,
                    "cluster_global_id": cs.global_id,
                    "pitch_group": cs.pitch_group,
                    "component": cs.component,
                    "draw_idx": k,
                    "LA": v2,
                    "SA": v3,
                    "xwOBAcon": v4,
                }
                if physics_calibrator is not None:
                    row["EV_dec"] = float(ed)
                    row["EV_cal"] = float(ec)
                    row["EV"] = float(ec)
                else:
                    row["EV"] = float(ed)
                rows_out.append(row)
            mean_ev_report = float(np.mean(ev_cal)) if ev_cal.size else float("nan")
            sum_rows.append(
                {
                    "batter_name": hitter,
                    "cluster_global_id": cs.global_id,
                    "pitch_group": cs.pitch_group,
                    "component": cs.component,
                    "n_admissible": int(dgi["n_admissible"]),
                    "n_attempted_decode_rows": int(dgi["n_attempted"]),
                    "converged": bool(dgi["converged"]),
                    "stopped_reason": dgi["stopped_reason"],
                    "n_soft_capped": int(dgi.get("n_soft_capped", 0)),
                    "max_EV_before": dgi.get("max_EV_before"),
                    "max_EV_after": dgi.get("max_EV_after"),
                    "p99_EV_before": dgi.get("p99_EV_before"),
                    "p99_EV_after": dgi.get("p99_EV_after"),
                    "mean_xwOBAcon": float(np.mean(xw)) if xw.size else float("nan"),
                    "std_xwOBAcon": float(np.std(xw, ddof=0)) if xw.size > 1 else float("nan"),
                    "mean_EV": mean_ev_report,
                    "mean_EV_dec": float(np.mean(ev_dec)) if ev_dec.size else float("nan"),
                    "mean_LA": float(np.mean(la)) if la.size else float("nan"),
                    "mean_SA": float(np.mean(sa)) if sa.size else float("nan"),
                }
            )
        df_long = pd.DataFrame(rows_out)
        df_sum = pd.DataFrame(sum_rows)
        safe = hitter.replace(" ", "_")
        df_long.to_parquet(results_parquet_dir / f"{safe}_robustness_draws.parquet", index=False)
        df_long.to_csv(results_parquet_dir / f"{safe}_robustness_draws.csv", index=False)
        df_sum.to_csv(summary_dir / f"{safe}_cluster_summary.csv", index=False)
        _save_hitter_distribution_png_robustness(
            df_long, results_parquet_dir / f"{safe}_distributions_ev_la_sa_value.png", title_slug=safe
        )

    print("Done. Outputs in", output_root)


if __name__ == "__main__":
    main()
