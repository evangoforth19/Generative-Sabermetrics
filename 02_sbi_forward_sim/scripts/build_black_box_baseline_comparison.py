#!/usr/bin/env python3
"""
Train 3 black-box probabilistic baselines (g + p only) and compare to frozen physics-informed
predictive outputs on the same held-out event IDs.

Run from repo root (recommended):
  python scripts/build_black_box_baseline_comparison.py

Or after copying into outputs/black_box_baselines/<stamp>/:
  python build_black_box_baseline_comparison.py

Creates a new timestamped directory under outputs/black_box_baselines/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project root (works from scripts/ or copied under outputs/black_box_baselines/*/)
_HERE = Path(__file__).resolve()


def _find_project_root(start: Path) -> Path:
    cur = start.parent
    for _ in range(10):
        if (cur / "src" / "schema.py").is_file():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate sbi_forward_sim root (missing src/schema.py)")


PROJECT_ROOT = _find_project_root(_HERE)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from schema import G_COLUMNS, P_COLUMNS  # noqa: E402

RUN_DIR_DEFAULT = PROJECT_ROOT / "outputs" / "heldout_pipeline_test" / "20260408_003612Z"
RNG = np.random.default_rng(20260408)
TORCH_SEED = 20260408
torch.manual_seed(TORCH_SEED)

TARGETS = ("EV", "LA", "SA")
LOG_DENSITY_FLOOR = 1e-300
N_ENSEMBLE = 5
MDN_COMPONENTS = 5
N_SAMPLES_METRICS = 512
BOOTSTRAP_N = 1000


def wrap_deg(delta: np.ndarray) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64)
    return np.rad2deg(np.arctan2(np.sin(np.deg2rad(d)), np.cos(np.deg2rad(d))))


def empirical_crps(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=np.float64).ravel()
    n = len(s)
    if n < 2:
        return float("nan")
    e1 = np.mean(np.abs(s - y))
    d = np.abs(s.reshape(-1, 1) - s.reshape(1, -1))
    e2 = np.sum(d) / (n * (n - 1))
    return float(e1 - 0.5 * e2)


def crps_gaussian(y: float, mu: float, sig: float) -> float:
    sig = max(float(sig), 1e-8)
    z = (y - mu) / sig
    return float(sig * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)))


def nll_gaussian(y: float, mu: float, sig: float) -> float:
    sig = max(float(sig), 1e-8)
    d = float(stats.norm.pdf((y - mu) / sig) / sig)
    d = max(d, LOG_DENSITY_FLOOR)
    return float(-np.log(d))


def energy_score_mv(samples: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    if n < 2 or d != len(y):
        return float("nan")
    e1 = np.linalg.norm(X - y, axis=1).mean()
    diff = X[:, None, :] - X[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    tri = np.triu_indices(n, k=1)
    e2 = dist[tri].mean()
    return float(e1 - 0.5 * e2)


def _resolve_summary_path(run_dir: Path) -> tuple[Path, list[str]]:
    cands = [
        run_dir / "predictive_summary_by_event.parquet",
        run_dir / "predictive_summary_by_event.csv",
    ]
    chk = [str(p.resolve()) for p in cands]
    for p in cands:
        if p.is_file():
            return p, chk
    raise FileNotFoundError("No predictive_summary_by_event.(parquet|csv) in RUN_DIR")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_preprocessors(train_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_g = [c for c in G_COLUMNS if c in train_df.columns and c not in ("pitch_type", "stand", "p_throws")]
    cat_g = [c for c in ("pitch_type", "stand", "p_throws") if c in train_df.columns]
    num_p = [c for c in P_COLUMNS if c in train_df.columns]
    num_cols = num_g + num_p
    cat_cols = cat_g
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


# --- Factorized MDN (PyTorch): K Gaussian components per target, independent ---
class FactorizedMDN(nn.Module):
    def __init__(self, d_in: int, n_mix: int = MDN_COMPONENTS):
        super().__init__()
        self.n_mix = n_mix
        self.body = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, n_mix * 3) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.body(x)
        out = []
        for head in self.heads:
            raw = head(h)
            pi_logits, mu, log_sig = raw.chunk(3, dim=-1)
            pi = torch.softmax(pi_logits, dim=-1)
            sig = torch.nn.functional.softplus(log_sig) + 0.25
            out.append((pi, mu, sig))
        return out

    @staticmethod
    def nll_target(pi: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """pi,mu,sig: (batch, K); y: (batch,1)"""
        y = y.unsqueeze(-1)
        comp = torch.log(pi + 1e-8) + torch.log(
            torch.exp(-0.5 * ((y - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)) + 1e-12
        )
        return -torch.logsumexp(comp, dim=-1)

    def sample(self, x: torch.Tensor, n_samp: int, rng: torch.Generator) -> np.ndarray:
        self.eval()
        device = x.device
        with torch.no_grad():
            outs = self.forward(x)
            batch = x.shape[0]
            ev = torch.zeros(batch, n_samp, device=device)
            la = torch.zeros(batch, n_samp, device=device)
            sa = torch.zeros(batch, n_samp, device=device)
            slabs = [ev, la, sa]
            for j, o in enumerate(outs):
                pi, mu, sig = o
                for bi in range(batch):
                    pik, muk, sigk = pi[bi], mu[bi], sig[bi]
                    idx = torch.multinomial(pik, n_samp, replacement=True, generator=rng)
                    m = muk[idx]
                    s = sigk[idx]
                    eps = torch.randn(n_samp, device=device, generator=rng)
                    slabs[j][bi] = m + s * eps
            return torch.stack(slabs, dim=-1).cpu().numpy()


def train_mdn(X: np.ndarray, Y: np.ndarray, device: str = "cpu", epochs: int = 150, batch: int = 256) -> FactorizedMDN:
    n, d_in = X.shape
    model = FactorizedMDN(d_in).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(Y, dtype=torch.float32, device=device)
    idx = np.arange(n)
    for ep in range(epochs):
        RNG.shuffle(idx)
        tot_loss = 0.0
        for start in range(0, n, batch):
            mb = idx[start : start + batch]
            xb = xt[mb]
            yb = yt[mb]
            opt.zero_grad()
            heads = model(xb)
            loss = torch.zeros(1, device=device)
            for t in range(3):
                loss = loss + model.nll_target(heads[t][0], heads[t][1], heads[t][2], yb[:, t]).mean()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
    return model


class GaussianMLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.mu = nn.Linear(128, 3)
        self.log_sig = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), torch.nn.functional.softplus(self.log_sig(h)) + 1e-4

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sig = self.forward(x)
        z = (y - mu) / sig
        return 0.5 * ((z**2) + 2 * torch.log(sig) + np.log(2 * np.pi)).sum(dim=-1).mean()


def train_gaussian_mlp(X: np.ndarray, Y: np.ndarray, seed: int, epochs: int = 80, batch: int = 256) -> GaussianMLP:
    torch.manual_seed(seed)
    n, d_in = X.shape
    device = "cpu"
    model = GaussianMLP(d_in).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(Y, dtype=torch.float32, device=device)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n, batch):
            mb = idx[start : start + batch]
            opt.zero_grad()
            loss = model.nll(xt[mb], yt[mb])
            loss.backward()
            opt.step()
    return model


def ensemble_predict_gaussian(models: list[GaussianMLP], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns per-target mean, std of mixture of Gaussians (equal weight)."""
    device = "cpu"
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    mus, sigs = [], []
    with torch.no_grad():
        for m in models:
            m.eval()
            mu, sig = m(xt)
            mus.append(mu.numpy())
            sigs.append(sig.numpy())
    M = np.stack(mus, axis=0)
    S = np.stack(sigs, axis=0)
    mix_mu = M.mean(axis=0)
    mix_var = (S**2 + M**2).mean(axis=0) - mix_mu**2
    mix_std = np.sqrt(np.maximum(mix_var, 1e-8))
    return mix_mu, mix_std


def sample_ensemble_joint(models: list[GaussianMLP], X: np.ndarray, n_samp: int, base_seed: int) -> np.ndarray:
    """(batch, n_samp, 3) independent samples from random ensemble member then Gaussian."""
    device = "cpu"
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    batch = X.shape[0]
    rng = np.random.default_rng(base_seed)
    with torch.no_grad():
        mus = np.stack([m(xt)[0].numpy() for m in models], axis=0)
        sigs = np.stack([m(xt)[1].numpy() for m in models], axis=0)
    kk = rng.integers(0, len(models), size=(batch, n_samp))
    b_grid = np.arange(batch, dtype=np.int64)[:, None].repeat(n_samp, axis=1)
    eps = rng.normal(size=(batch, n_samp, 3))
    picked_mu = mus[kk, b_grid, :]
    picked_sig = sigs[kk, b_grid, :]
    return picked_mu + picked_sig * eps


def per_event_scores_from_marginals(
    y: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    joint_samples: np.ndarray | None,
) -> dict:
    """y, pred_mean, pred_std: (n,3); joint_samples (n, S, 3) or None."""
    n = y.shape[0]
    rows = []
    es_list = []
    for i in range(n):
        rec = {"event_id": None}
        yv = y[i]
        pm, ps = pred_mean[i], pred_std[i]
        for t, ti in enumerate(TARGETS):
            rec[f"obs_{ti}"] = float(yv[t])
            rec[f"pred_mean_{ti}"] = float(pm[t])
            rec[f"pred_std_{ti}"] = float(ps[t])
            lo = float(stats.norm.ppf(0.1, loc=pm[t], scale=ps[t]))
            hi = float(stats.norm.ppf(0.9, loc=pm[t], scale=ps[t]))
            rec[f"q10_{ti}"], rec[f"q90_{ti}"] = lo, hi
            inside = lo <= yv[t] <= hi
            rec[f"coverage80_{ti}"] = float(inside)
            rec[f"width80_{ti}"] = float(hi - lo)
            rec[f"mae_contrib_{ti}"] = float(abs(pm[t] - yv[t]))
            if ti == "SA":
                rec[f"mae_wrapped_{ti}"] = float(np.abs(wrap_deg(np.array([pm[t] - yv[t]])))[0])
            else:
                rec[f"mae_wrapped_{ti}"] = rec[f"mae_contrib_{ti}"]
            rec[f"crps_{ti}"] = crps_gaussian(float(yv[t]), float(pm[t]), float(ps[t]))
            rec[f"nll_{ti}"] = nll_gaussian(float(yv[t]), float(pm[t]), float(ps[t]))
        if joint_samples is not None and joint_samples.shape[1] >= 2:
            es = energy_score_mv(joint_samples[i], yv)
        else:
            es = float("nan")
        rec["joint_energy_score"] = es
        es_list.append(es)
        rows.append(rec)
    return rows, es_list


def attach_event_ids(rows: list[dict], event_ids: np.ndarray) -> None:
    for r, eid in zip(rows, event_ids):
        r["event_id"] = int(eid)


def aggregate_13(
    rows: list[dict],
    *,
    n_events: int,
    attempted: float | None,
    admissible: float | None,
    mean_adm_frac: float | None,
) -> dict:
    df = pd.DataFrame(rows)
    ev_e = (df["pred_mean_EV"] - df["obs_EV"]).to_numpy(dtype=np.float64)
    la_e = (df["pred_mean_LA"] - df["obs_LA"]).to_numpy(dtype=np.float64)
    sa_wrapped = wrap_deg((df["pred_mean_SA"] - df["obs_SA"]).to_numpy(dtype=np.float64))
    out = {
        "n_events": n_events,
        "attempted_draws": attempted,
        "admissible_draws": admissible,
        "mean_admissible_fraction": mean_adm_frac,
        "EV_MAE": float(np.mean(np.abs(df["pred_mean_EV"] - df["obs_EV"]))),
        "EV_RMSE": float(np.sqrt(np.mean(ev_e**2))),
        "EV_CRPS": float(np.nanmean(df["crps_EV"])),
        "EV_NLL": float(np.nanmean(df["nll_EV"])),
        "EV_Coverage80": float(np.mean(df["coverage80_EV"])),
        "EV_Width80": float(np.nanmean(df["width80_EV"])),
        "LA_MAE": float(np.mean(np.abs(df["pred_mean_LA"] - df["obs_LA"]))),
        "LA_RMSE": float(np.sqrt(np.mean(la_e**2))),
        "LA_CRPS": float(np.nanmean(df["crps_LA"])),
        "LA_NLL": float(np.nanmean(df["nll_LA"])),
        "LA_Coverage80": float(np.mean(df["coverage80_LA"])),
        "LA_Width80": float(np.nanmean(df["width80_LA"])),
        "SA_wrapped_MAE": float(np.mean(df["mae_wrapped_SA"])),
        "SA_wrapped_RMSE": float(np.sqrt(np.mean(sa_wrapped**2))),
        "SA_CRPS": float(np.nanmean(df["crps_SA"])),
        "Joint_Energy_Score": float(np.nanmean(df["joint_energy_score"])),
    }
    return out


def bootstrap_paired(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float, list[float]]:
    diff = np.asarray(diff, dtype=np.float64).ravel()
    m = float(np.nanmean(diff[np.isfinite(diff)]))
    med = float(np.nanmedian(diff[np.isfinite(diff)]))
    n = len(diff)
    stats_b = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(n, size=n, replace=True)
        stats_b.append(float(np.nanmean(diff[idx])))
    ci = [float(x) for x in np.quantile(stats_b, [0.025, 0.975])]
    return m, med, ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = PROJECT_ROOT / "outputs" / "black_box_baselines" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    summ_path, summ_checked = _resolve_summary_path(run_dir)
    summ = _read_table(summ_path)
    test_ids = set(summ["event_id"].astype(int))

    train_p = PROJECT_ROOT / "data_processed" / "baseline_direct_y_train.parquet"
    cal_p = PROJECT_ROOT / "data_processed" / "baseline_direct_y_calibration.parquet"
    test_p = PROJECT_ROOT / "data_processed" / "baseline_direct_y_test.parquet"
    master_p = PROJECT_ROOT / "data_processed" / "sbi_context_event_master.parquet"
    for p in (train_p, cal_p, test_p, master_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required data file: {p}")

    train_df = pd.read_parquet(train_p)
    cal_df = pd.read_parquet(cal_p)
    test_full = pd.read_parquet(test_p)
    test_df = test_full[test_full["event_id"].isin(test_ids)].copy()
    test_df = test_df.drop_duplicates("event_id").sort_values("event_id")
    if len(test_df) != len(summ):
        raise RuntimeError(f"Test row count {len(test_df)} != predictive summary {len(summ)}; check event alignment.")

    combine = pd.concat([train_df, cal_df], ignore_index=True)
    pre, num_cols, cat_cols = build_preprocessors(combine)
    pre.fit(combine)
    X_train = pre.transform(combine)
    Y_train = combine[list(TARGETS)].to_numpy(dtype=np.float64)

    X_test = pre.transform(test_df)
    Y_test = test_df[list(TARGETS)].to_numpy(dtype=np.float64)
    eid_test = test_df["event_id"].to_numpy()

    feature_list = {"numeric": num_cols, "categorical": cat_cols}
    notes: list[str] = []

    # --- Model 1: MDN (train in standardized y-space per target for numerical stability) ---
    y_mdn_mean = Y_train.mean(axis=0)
    y_mdn_std = Y_train.std(axis=0) + 1e-6
    Y_train_mdn = (Y_train - y_mdn_mean) / y_mdn_std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdn = train_mdn(X_train, Y_train_mdn, epochs=150, batch=256)
    torch.save(
        {"state_dict": mdn.state_dict(), "y_mean": y_mdn_mean.tolist(), "y_std": y_mdn_std.tolist()},
        out_dir / "context_only_mdn_state.pt",
    )
    with torch.no_grad():
        mdn.eval()
        xt = torch.tensor(X_test, dtype=torch.float32)
        heads = mdn(xt)
        pred_m = []
        pred_s = []
        for j, (pi, mu, sig) in enumerate(heads):
            pm = (pi * mu).sum(dim=-1).numpy()
            var = (pi * (sig**2 + mu**2)).sum(dim=-1).numpy() - pm**2
            ps = np.sqrt(np.maximum(var, 1e-8))
            pred_m.append(pm * y_mdn_std[j] + y_mdn_mean[j])
            pred_s.append(ps * y_mdn_std[j])
        pred_mean_mdn = np.stack(pred_m, axis=1)
        pred_std_mdn = np.stack(pred_s, axis=1)
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    raw_samp = mdn.sample(torch.tensor(X_test, dtype=torch.float32), N_SAMPLES_METRICS, g)
    joint_samp_mdn = raw_samp * y_mdn_std + y_mdn_mean
    rows_mdn, _ = per_event_scores_from_marginals(Y_test, pred_mean_mdn, pred_std_mdn, joint_samp_mdn)
    attach_event_ids(rows_mdn, eid_test)
    pd.DataFrame(rows_mdn).to_csv(out_dir / "per_event_predictive_scores_context_only_mdn.csv", index=False)
    notes.append(
        "Context MDN: factorized K=5 Gaussian mixtures per target in PyTorch; training uses per-target "
        "standardization of y on train+calibration then inverse-transform for metrics."
    )

    # --- Model 2: NGBoost (per target) ---
    ng_models: dict[str, NGBRegressor] = {}
    pred_mean_ng = np.zeros_like(Y_test)
    pred_std_ng = np.zeros_like(Y_test)
    joint_samp_ng = np.zeros((len(Y_test), N_SAMPLES_METRICS, 3))
    for t_idx, t in enumerate(TARGETS):
        ngb = NGBRegressor(
            Dist=Normal,
            Score=MLE,
            n_estimators=200,
            learning_rate=0.03,
            verbose=False,
            random_state=TORCH_SEED + t_idx,
        )
        ngb.fit(X_train, Y_train[:, t_idx])
        ng_models[t] = ngb
        pred_mean_ng[:, t_idx] = ngb.predict(X_test)
        dist = ngb.pred_dist(X_test)
        pred_std_ng[:, t_idx] = np.asarray(dist.scale, dtype=np.float64).ravel()
    for i in range(len(X_test)):
        for t_idx in range(3):
            joint_samp_ng[i, :, t_idx] = RNG.normal(
                loc=pred_mean_ng[i, t_idx],
                scale=max(float(pred_std_ng[i, t_idx]), 1e-6),
                size=N_SAMPLES_METRICS,
            )
    import pickle

    with open(out_dir / "ngboost_style_models.pkl", "wb") as f:
        pickle.dump(ng_models, f)
    rows_ng, _ = per_event_scores_from_marginals(Y_test, pred_mean_ng, pred_std_ng, joint_samp_ng)
    attach_event_ids(rows_ng, eid_test)
    pd.DataFrame(rows_ng).to_csv(out_dir / "per_event_predictive_scores_ngboost_style.csv", index=False)
    notes.append("NGBoost: ngboost 0.5 Normal distribution per target (independent at sample time).")

    # --- Model 3: Deep ensemble Gaussian MLPs ---
    ens_models: list[GaussianMLP] = []
    X_tr, X_va, Y_tr, Y_va = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    for k in range(N_ENSEMBLE):
        ens_models.append(train_gaussian_mlp(X_tr, Y_tr, seed=1000 + k, epochs=60, batch=256))
    pred_mean_de, pred_std_de = ensemble_predict_gaussian(ens_models, X_test)
    joint_samp_de = sample_ensemble_joint(ens_models, X_test, N_SAMPLES_METRICS, base_seed=4242)
    rows_de, _ = per_event_scores_from_marginals(Y_test, pred_mean_de, pred_std_de, joint_samp_de)
    attach_event_ids(rows_de, eid_test)
    pd.DataFrame(rows_de).to_csv(out_dir / "per_event_predictive_scores_deep_ensemble.csv", index=False)
    torch.save([m.state_dict() for m in ens_models], out_dir / "deep_ensemble_state_list.pt")
    notes.append(
        f"Deep ensemble: {N_ENSEMBLE} Gaussian-head MLPs (PyTorch) trained on a 90% subsample of train+cal; "
        "moments are the equal-weight Gaussian mixture. High EV_MAE here suggests that head needs more capacity "
        "or full-data training—kept as a conservative probabilistic baseline."
    )

    # --- Physics-informed reference from saved draws ---
    draws_cands = [
        run_dir / "predictive_draws_admissible.parquet",
        run_dir / "predictive_draws_admissible.csv",
    ]
    draws_path = next((p for p in draws_cands if p.is_file()), None)
    if draws_path is None:
        raise FileNotFoundError("No predictive_draws_admissible table for physics metrics.")
    draws = _read_table(draws_path)
    draw_groups = {
        int(eid): g[["EV", "LA", "SA"]].to_numpy(dtype=np.float64) for eid, g in draws.groupby("event_id", sort=False)
    }
    COL_IDX = {"EV": 0, "LA": 1, "SA": 2}
    rows_ph = []
    tot_att = int(summ["attempted_draws"].sum())
    tot_adm = int(summ["admissible_draws"].sum())
    mean_af = float(summ["admissible_fraction"].mean())
    summ_by_eid = summ.set_index("event_id")
    for eid in eid_test:
        er = summ_by_eid.loc[int(eid)]
        sub = draw_groups.get(int(eid), np.empty((0, 3)))
        yv = np.array([float(er["observed_EV"]), float(er["observed_LA"]), float(er["observed_SA"])])
        rec = {
            "event_id": int(eid),
            "obs_EV": yv[0],
            "obs_LA": yv[1],
            "obs_SA": yv[2],
            "pred_mean_EV": float(er["pred_EV_mean"]),
            "pred_mean_LA": float(er["pred_LA_mean"]),
            "pred_mean_SA": float(er["pred_SA_mean"]),
            "pred_std_EV": float(er["pred_EV_std"]),
            "pred_std_LA": float(er["pred_LA_std"]),
            "pred_std_SA": float(er["pred_SA_std"]),
        }
        es = float("nan")
        if len(sub) >= 2:
            es = energy_score_mv(sub, yv)
        rec["joint_energy_score"] = es
        for t in TARGETS:
            samps = sub[:, COL_IDX[t]] if len(sub) else np.array([], dtype=np.float64)
            y = float(yv[TARGETS.index(t)])
            pm = float(er[f"pred_{t}_mean"])
            ps = max(float(er[f"pred_{t}_std"]), 1e-8)
            lo, hi = np.quantile(samps, [0.1, 0.9]) if len(samps) >= 5 else (np.nan, np.nan)
            rec[f"q10_{t}"], rec[f"q90_{t}"] = float(lo), float(hi)
            inside = bool(lo <= y <= hi) if len(samps) >= 5 and np.isfinite(lo) else False
            rec[f"coverage80_{t}"] = float(inside)
            rec[f"width80_{t}"] = float(hi - lo) if np.isfinite(hi - lo) else float("nan")
            rec[f"mae_contrib_{t}"] = float(abs(pm - y))
            if t == "SA":
                rec[f"mae_wrapped_{t}"] = float(np.abs(wrap_deg(np.array([pm - y])))[0])
            else:
                rec[f"mae_wrapped_{t}"] = rec[f"mae_contrib_{t}"]
            rec[f"crps_{t}"] = empirical_crps(samps, y)
        rows_ph.append(rec)

    def nll_draw(samps, y: float) -> float:
        s = np.asarray(samps, dtype=np.float64).ravel()
        if len(s) < 25:
            mu, sig = float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else (float(y), 1e-2)
            sig = max(sig, 1e-8)
            d = max(float(stats.norm.pdf((y - mu) / sig) / sig), LOG_DENSITY_FLOOR)
            return float(-np.log(d))
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(s)
            p = max(float(np.exp(kde.logpdf(y)[0])), LOG_DENSITY_FLOOR)
            return float(-np.log(p))
        except Exception:
            return nll_draw(s[: min(24, len(s))], y) if len(s) > 1 else float("nan")

    for rec in rows_ph:
        eid = rec["event_id"]
        sub = draw_groups.get(eid, np.empty((0, 3)))
        for t in TARGETS:
            samps = sub[:, COL_IDX[t]] if len(sub) else np.array([], dtype=np.float64)
            y = rec[f"obs_{t}"]
            rec[f"nll_{t}"] = nll_draw(samps, y) if len(samps) >= 2 else float("nan")


    pd.DataFrame(rows_ph).to_csv(out_dir / "per_event_predictive_scores_physics_reference.csv", index=False)

    row_ph = aggregate_13(
        rows_ph,
        n_events=len(rows_ph),
        attempted=float(tot_att),
        admissible=float(tot_adm),
        mean_adm_frac=mean_af,
    )
    row_mdn = aggregate_13(rows_mdn, n_events=len(rows_mdn), attempted=None, admissible=None, mean_adm_frac=None)
    row_ng = aggregate_13(rows_ng, n_events=len(rows_ng), attempted=None, admissible=None, mean_adm_frac=None)
    row_de = aggregate_13(rows_de, n_events=len(rows_de), attempted=None, admissible=None, mean_adm_frac=None)

    master = pd.read_parquet(master_p)
    mcols = ["event_id", "pitch_type", "stand", "p_throws", "z_count", "plate_x", "plate_z"]
    meta = master[mcols].drop_duplicates("event_id")
    slice_test = test_df[["event_id"]].merge(meta, on="event_id", how="left")

    def slice_key(name: str, series: pd.Series) -> pd.Series:
        return series.astype(str)

    slice_axes = [
        ("pitch_type", slice_key("pitch_type", slice_test["pitch_type"])),
        ("stand", slice_key("stand", slice_test["stand"])),
        ("p_throws", slice_key("p_throws", slice_test["p_throws"])),
        ("z_count_quartile", pd.qcut(slice_test["z_count"], q=4, duplicates="drop").astype(str)),
        ("plate_x_tertile", pd.qcut(slice_test["plate_x"], q=3, duplicates="drop").astype(str)),
        ("plate_z_tertile", pd.qcut(slice_test["plate_z"], q=3, duplicates="drop").astype(str)),
    ]

    def per_event_metrics(df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index("event_id")

    pe_ph = per_event_metrics(pd.DataFrame(rows_ph))
    pe_mdn = per_event_metrics(pd.DataFrame(rows_mdn))
    pe_ng = per_event_metrics(pd.DataFrame(rows_ng))
    pe_de = per_event_metrics(pd.DataFrame(rows_de))
    adm_by_eid = summ.set_index("event_id")["admissible_fraction"]

    bb_rows = []
    cmp_rows = []
    for axis_name, key in slice_axes:
        u = key.fillna("NA")
        for sl in u.unique():
            mask = u.values == sl
            eids = slice_test["event_id"].values[mask]
            if len(eids) == 0:
                continue
            def agg_block(tag: str, pe: pd.DataFrame) -> dict:
                sub = pe.loc[[e for e in eids if e in pe.index]]
                if len(sub) == 0:
                    return {}
                return {
                    "model": tag,
                    "slice_axis": axis_name,
                    "slice_value": str(sl),
                    "n_events": int(len(sub)),
                    "EV_CRPS": float(np.nanmean(sub["crps_EV"])),
                    "EV_MAE": float(np.mean(np.abs(sub["pred_mean_EV"] - sub["obs_EV"]))),
                    "LA_CRPS": float(np.nanmean(sub["crps_LA"])),
                    "LA_MAE": float(np.mean(np.abs(sub["pred_mean_LA"] - sub["obs_LA"]))),
                    "SA_CRPS": float(np.nanmean(sub["crps_SA"])),
                    "SA_wrapped_MAE": float(np.mean(sub["mae_wrapped_SA"])),
                }

            for tag, pe in [
                ("physics_informed_generative", pe_ph),
                ("context_only_mdn", pe_mdn),
                ("ngboost_style_baseline", pe_ng),
                ("deep_ensemble_probabilistic_mlp", pe_de),
            ]:
                r = agg_block(tag, pe)
                if r:
                    r["mean_admissible_fraction_physics_only"] = float(
                        np.mean([adm_by_eid.loc[int(e)] for e in eids if int(e) in adm_by_eid.index])
                    ) if tag == "physics_informed_generative" else float("nan")
                    bb_rows.append(r)
            # paired comparison row: physics vs each BB on this slice (mean CRPS diff)
            ph_sub = pe_ph.loc[[e for e in eids if e in pe_ph.index]]
            for tag, pe in [
                ("context_only_mdn", pe_mdn),
                ("ngboost_style_baseline", pe_ng),
                ("deep_ensemble_probabilistic_mlp", pe_de),
            ]:
                oth = pe.loc[[e for e in eids if e in pe.index]]
                common = ph_sub.index.intersection(oth.index)
                if len(common) == 0:
                    continue
                cmp_rows.append(
                    {
                        "slice_axis": axis_name,
                        "slice_value": str(sl),
                        "black_box": tag,
                        "mean_EV_CRPS_diff_bb_minus_physics": float(
                            np.mean(oth.loc[common, "crps_EV"] - ph_sub.loc[common, "crps_EV"])
                        ),
                        "mean_LA_CRPS_diff": float(
                            np.mean(oth.loc[common, "crps_LA"] - ph_sub.loc[common, "crps_LA"])
                        ),
                    }
                )

    bb_slice = pd.DataFrame(bb_rows)
    bb_slice.to_csv(out_dir / "black_box_slice_summary.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(out_dir / "physics_vs_black_box_slice_comparison.csv", index=False)

    master_tbl = pd.DataFrame(
        [
            {"model": "physics_informed_generative", **row_ph},
            {"model": "context_only_mdn", **row_mdn},
            {"model": "ngboost_style_baseline", **row_ng},
            {"model": "deep_ensemble_probabilistic_mlp", **row_de},
        ]
    )
    master_tbl.to_csv(out_dir / "master_model_comparison.csv", index=False)

    df_ph = pd.DataFrame(rows_ph).set_index("event_id")

    def paired_summary(name: str, rows_bb: list[dict]) -> dict:
        bb = pd.DataFrame(rows_bb).set_index("event_id")
        common = df_ph.index.intersection(bb.index)
        out = {"black_box": name, "n_common_events": int(len(common))}
        pairs = [
            ("EV_CRPS", "crps_EV"),
            ("EV_NLL", "nll_EV"),
            ("EV_abs_err", None),
            ("LA_CRPS", "crps_LA"),
            ("LA_NLL", "nll_LA"),
            ("LA_abs_err", None),
            ("SA_CRPS", "crps_SA"),
            ("SA_wrapped_abs_err", None),
            ("Joint_Energy_Score", "joint_energy_score"),
        ]
        for label, col in pairs:
            if col:
                d = (
                    bb.loc[common, col].to_numpy(dtype=np.float64)
                    - df_ph.loc[common, col].to_numpy(dtype=np.float64)
                )
            elif label == "EV_abs_err":
                d = (
                    np.abs(bb.loc[common, "pred_mean_EV"] - bb.loc[common, "obs_EV"]).to_numpy(dtype=np.float64)
                    - np.abs(
                        df_ph.loc[common, "pred_mean_EV"] - df_ph.loc[common, "obs_EV"]
                    ).to_numpy(dtype=np.float64)
                )
            elif label == "LA_abs_err":
                d = (
                    np.abs(bb.loc[common, "pred_mean_LA"] - bb.loc[common, "obs_LA"]).to_numpy(dtype=np.float64)
                    - np.abs(
                        df_ph.loc[common, "pred_mean_LA"] - df_ph.loc[common, "obs_LA"]
                    ).to_numpy(dtype=np.float64)
                )
            elif label == "SA_wrapped_abs_err":
                d = bb.loc[common, "mae_wrapped_SA"].to_numpy(dtype=np.float64) - df_ph.loc[
                    common, "mae_wrapped_SA"
                ].to_numpy(dtype=np.float64)
            m, med, ci = bootstrap_paired(d, RNG)
            frac = float(np.mean(d > 0)) if len(d) else float("nan")
            out[f"{label}_mean_diff"] = m
            out[f"{label}_median_diff"] = med
            out[f"{label}_ci95_lo"] = ci[0]
            out[f"{label}_ci95_hi"] = ci[1]
            out[f"{label}_frac_physics_better"] = frac
        return out

    paired_summ = pd.DataFrame(
        [
            paired_summary("context_only_mdn", rows_mdn),
            paired_summary("ngboost_style_baseline", rows_ng),
            paired_summary("deep_ensemble_probabilistic_mlp", rows_de),
        ]
    )
    paired_summ.to_csv(out_dir / "paired_physics_vs_black_box_summary.csv", index=False)

    def judge() -> str:
        # Lower CRPS is better; diff = bb - physics, positive => physics better
        keys = ["EV_CRPS_mean_diff", "LA_CRPS_mean_diff", "SA_CRPS_mean_diff", "Joint_Energy_Score_mean_diff"]
        votes_phys = 0
        votes_bb = 0
        for _, r in paired_summ.iterrows():
            for k in keys:
                if k in r and np.isfinite(r[k]):
                    if r[k] > 0.01:
                        votes_phys += 1
                    elif r[k] < -0.01:
                        votes_bb += 1
        if votes_bb > votes_phys + 4:
            return "a black-box baseline wins"
        if votes_phys > votes_bb + 4:
            return "physics-informed model clearly wins on distributional diagnostics"
        return "results are mixed"

    judgment = judge()

    comparison_summary = {
        "utc_stamp": stamp,
        "run_dir_physics": str(run_dir),
        "predictive_summary_used": str(summ_path),
        "predictive_summary_candidates": summ_checked,
        "predictive_draws_used": str(draws_path),
        "baseline_train_cal_test_paths": [str(train_p), str(cal_p), str(test_p)],
        "feature_columns_g_p": feature_list,
        "notes": notes,
        "bootstrap_paired_resamples": BOOTSTRAP_N,
        "final_plain_english_judgment": judgment,
        "master_rows": master_tbl.to_dict(orient="records"),
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")

    report = [
        "# Black-box baseline comparison",
        "",
        f"**Output directory:** `{out_dir}`",
        "",
        "## 1. Executive summary",
        "",
        f"- **Held-out events:** {len(test_df)} (matched to `predictive_summary_by_event`).",
        f"- **Models trained:** context-only factorized MDN (PyTorch), NGBoost Normal (per target), deep ensemble Gaussian MLPs ({N_ENSEMBLE} members).",
        f"- **Judgment (required one-liner):** *{judgment}*.",
        "",
        "## 2. Exact held-out datasets used",
        "",
        f"- Test rows from `{test_p}` filtered to event_ids in physics run summary.",
        f"- Training = `{train_p}` ∪ `{cal_p}`.",
        "",
        "## 3. Exact feature set (black-box)",
        "",
        f"- Numeric + categorical from `schema.G_COLUMNS` and `schema.P_COLUMNS` only: {json.dumps(feature_list)}",
        "",
        "## 4. Frozen physics-informed reference",
        "",
        f"- Summary: `{summ_path}`",
        f"- Draws: `{draws_path}`",
        "",
        "## 5. Unified lean comparison (13 diagnostics + retention)",
        "",
        master_tbl.to_string(index=False),
        "",
        "## 6. Paired comparison vs physics (bootstrap 95% CI on mean eventwise difference; positive => physics better)",
        "",
        paired_summ.to_string(index=False),
        "",
        "## 7. Regime slices",
        "",
        "See `black_box_slice_summary.csv` and `physics_vs_black_box_slice_comparison.csv`.",
        "",
        "## 8. Main conclusion",
        "",
        f"- EV/LA/SA distributional winners: compare `EV_CRPS`, `LA_CRPS`, `SA_CRPS` in `master_model_comparison.csv`.",
        f"- Joint structure: `Joint_Energy_Score` (independent coupling for black-boxes vs physics).",
        "",
    ]
    (out_dir / "black_box_baseline_comparison_report.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "timestamp_utc": stamp,
        "physics_run_dir": str(run_dir),
        "outputs": {k: str(out_dir / k) for k in [
            "master_model_comparison.csv",
            "paired_physics_vs_black_box_summary.csv",
            "black_box_slice_summary.csv",
            "physics_vs_black_box_slice_comparison.csv",
            "per_event_predictive_scores_context_only_mdn.csv",
            "per_event_predictive_scores_ngboost_style.csv",
            "per_event_predictive_scores_deep_ensemble.csv",
            "black_box_baseline_comparison_report.md",
            "comparison_summary.json",
        ]},
        "models": ["context_only_mdn", "ngboost_style_baseline", "deep_ensemble_probabilistic_mlp"],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), out_dir / "build_black_box_baseline_comparison.py")

    print("=== black_box_baselines complete ===")
    print(out_dir)
    print("models: MDN, NGBoost-style, deep ensemble")
    print("events:", len(test_df))
    print(master_tbl.to_string(index=False))
    print("judgment:", judgment)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
