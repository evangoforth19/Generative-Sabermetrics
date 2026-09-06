"""Stage-u output heads: full 3D Gaussian and hybrid (v_ss,a) Gaussian × d GMM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import VonMises


class GaussianCholeskyNet(nn.Module):
    """
    Outputs μ ∈ R³ and lower-triangular Cholesky factor L with Σ = L Lᵀ (SPD).

    L entries order: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
    Diagonal via softplus + eps for positivity.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        chol_eps: float = 1e-5,
    ):
        super().__init__()
        self.chol_eps = chol_eps
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation.lower()]

        emb_sum = sum(embedding_dims[k] for k in sorted(embedding_dims))
        layers: list[nn.Module] = []
        dim = input_dim + emb_sum
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 9)  # 3 mean + 6 chol

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        embs = [self.embeddings[k](cat[k]) for k in sorted(cat.keys())]
        h = self.mlp(torch.cat([x_num] + embs, dim=-1))
        out = self.head(h)
        mu = out[:, :3]
        raw = out[:, 3:]
        l00 = torch.nn.functional.softplus(raw[:, 0]) + self.chol_eps
        l10 = raw[:, 1]
        l11 = torch.nn.functional.softplus(raw[:, 2]) + self.chol_eps
        l20 = raw[:, 3]
        l21 = raw[:, 4]
        l22 = torch.nn.functional.softplus(raw[:, 5]) + self.chol_eps
        zeros = torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)
        row0 = torch.stack([l00, zeros, zeros], dim=-1)
        row1 = torch.stack([l10, l11, zeros], dim=-1)
        row2 = torch.stack([l20, l21, l22], dim=-1)
        tril = torch.stack([row0, row1, row2], dim=-2)
        return mu, tril


def log_prob_standard_normal(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (z * z + math.log(2 * math.pi))


def mixture_log_prob_1d(
    y: torch.Tensor,
    mix_logits: torch.Tensor,
    mix_mu: torch.Tensor,
    mix_scale: torch.Tensor,
    *,
    log_weights_eps: float = 1e-8,
    scale_eps: float = 1e-8,
) -> torch.Tensor:
    """
    Log-density of Σ_k π_k N(y | μ_k, σ_k²) with π = softmax(mix_logits).

    Shapes: y (B,), mix_* (B, K).
    """
    pi = torch.nn.functional.softmax(mix_logits, dim=-1).clamp_min(log_weights_eps)
    log_pi = torch.log(pi)
    y_e = y.unsqueeze(-1)
    sigma = mix_scale.clamp_min(scale_eps)
    z = (y_e - mix_mu) / sigma
    log_comp = log_prob_standard_normal(z) - torch.log(sigma)
    return torch.logsumexp(log_pi + log_comp, dim=-1)


def mixture_cdf_1d(
    y: torch.Tensor,
    mix_logits: torch.Tensor,
    mix_mu: torch.Tensor,
    mix_scale: torch.Tensor,
    *,
    scale_eps: float = 1e-8,
) -> torch.Tensor:
    """CDF F(y) = Σ_k π_k Φ((y - μ_k) / σ_k); y (B,), mix_* (B,K)."""
    pi = torch.nn.functional.softmax(mix_logits, dim=-1)
    y_e = y.unsqueeze(-1)
    sigma = mix_scale.clamp_min(scale_eps)
    z = (y_e - mix_mu) / sigma
    return (pi * torch.special.ndtr(z)).sum(dim=-1)


def mixture_marginal_mean_var(
    mix_logits: torch.Tensor,
    mix_mu: torch.Tensor,
    mix_scale: torch.Tensor,
    *,
    scale_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Law of total variance: E[d], Var[d] for a Gaussian mixture (B,) each."""
    pi = torch.nn.functional.softmax(mix_logits, dim=-1)
    sig2 = mix_scale.clamp_min(scale_eps) ** 2
    mean_mix = (pi * mix_mu).sum(dim=-1)
    ex2 = (pi * (sig2 + mix_mu * mix_mu)).sum(dim=-1)
    var_mix = (ex2 - mean_mix * mean_mix).clamp_min(scale_eps**2)
    return mean_mix, var_mix


class HybridBivariateGaussianMixtureDNet(nn.Module):
    """
    p(v_ss,a,d|x) = N(v_ss,a | μ, Σ) · Σ_k π_k N(d | μ_k, σ_k²).

    (v_ss,a) use a 2×2 Cholesky factor; d uses K Gaussian components on standardized
    ``d_tilde`` scale (same as training targets y[:,2]).
    """

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_mixture: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        chol_eps: float = 1e-5,
    ):
        super().__init__()
        if n_mixture < 1:
            raise ValueError("n_mixture must be >= 1")
        self.n_mixture = int(n_mixture)
        self.chol_eps = float(chol_eps)
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation.lower()]
        emb_sum = sum(embedding_dims[k] for k in sorted(embedding_dims))
        layers: list[nn.Module] = []
        dim = input_dim + emb_sum
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        head_out = 2 + 3 + 3 * self.n_mixture
        self.head = nn.Linear(dim, head_out)

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        embs = [self.embeddings[k](cat[k]) for k in sorted(cat.keys())]
        h = self.mlp(torch.cat([x_num] + embs, dim=-1))
        out = self.head(h)
        mu_va = out[:, :2]
        raw = out[:, 2:]
        l00 = torch.nn.functional.softplus(raw[:, 0]) + self.chol_eps
        l10 = raw[:, 1]
        l11 = torch.nn.functional.softplus(raw[:, 2]) + self.chol_eps
        z = torch.zeros(mu_va.shape[0], device=mu_va.device, dtype=mu_va.dtype)
        row0 = torch.stack([l00, z], dim=-1)
        row1 = torch.stack([l10, l11], dim=-1)
        tril_va = torch.stack([row0, row1], dim=-2)

        k = self.n_mixture
        logits = raw[:, 3 : 3 + k]
        mu_m = raw[:, 3 + k : 3 + 2 * k]
        sig = torch.nn.functional.softplus(raw[:, 3 + 2 * k : 3 + 3 * k]) + self.chol_eps
        return mu_va, tril_va, logits, mu_m, sig

    def joint_log_prob(
        self,
        y: torch.Tensor,
        mu_va: torch.Tensor,
        tril_va: torch.Tensor,
        mix_logits: torch.Tensor,
        mix_mu: torch.Tensor,
        mix_scale: torch.Tensor,
    ) -> torch.Tensor:
        """y (B,3) standardized; returns log p (B,)."""
        dist2 = torch.distributions.MultivariateNormal(
            mu_va, scale_tril=tril_va, validate_args=False
        )
        lp2 = dist2.log_prob(y[:, :2])
        lpd = mixture_log_prob_1d(y[:, 2], mix_logits, mix_mu, mix_scale)
        return lp2 + lpd


def wrap_angle_rad(rad: torch.Tensor) -> torch.Tensor:
    """Wrap radians to (-π, π]."""
    return torch.atan2(torch.sin(rad), torch.cos(rad))


def shared_mixture_bivariate_gaussian_vonmises_log_prob(
    y_va_std: torch.Tensor,
    d_rad: torch.Tensor,
    mix_logits: torch.Tensor,
    mu_va: torch.Tensor,
    L_va: torch.Tensor,
    loc_d: torch.Tensor,
    kappa_d: torch.Tensor,
    *,
    kappa_max: float = 120.0,
) -> torch.Tensor:
    """
    log sum_k pi_k N(y_va | μ_k, L_k L_k^T) VM(d | loc_k, κ_k).

    y_va_std: (B, 2) standardized; d_rad: (B,) radians (any real, wrapped internally).
    mu_va: (B, K, 2); L_va: (B, K, 2, 2); loc_d, kappa_d: (B, K).
    """
    d_w = wrap_angle_rad(d_rad)
    log_pi = F.log_softmax(mix_logits, dim=-1)
    kappa_c = kappa_d.clamp(min=1e-4, max=float(kappa_max))
    K = mix_logits.shape[-1]
    lp_g_rows = []
    lp_v_rows = []
    for k in range(K):
        dist2 = torch.distributions.MultivariateNormal(
            mu_va[:, k, :],
            scale_tril=L_va[:, k, :, :],
            validate_args=False,
        )
        lp_g_rows.append(dist2.log_prob(y_va_std))
        dist_v = VonMises(loc_d[:, k], kappa_c[:, k], validate_args=False)
        lp_v_rows.append(dist_v.log_prob(d_w))
    lp_g = torch.stack(lp_g_rows, dim=1)
    lp_v = torch.stack(lp_v_rows, dim=1)
    return torch.logsumexp(log_pi + lp_g + lp_v, dim=-1)


class SharedMixtureGaussianVonMisesUNet(nn.Module):
    """
    K-component mixture with **shared** π_k(x): 2D Gaussian on standardized (v_ss, a)
    and von Mises on d_tilde in radians (from raw degrees).
    """

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_mixture: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        chol_eps: float = 1e-5,
        kappa_floor: float = 1e-3,
        kappa_max: float = 120.0,
    ):
        super().__init__()
        if n_mixture < 1:
            raise ValueError("n_mixture must be >= 1")
        self.n_mixture = int(n_mixture)
        self.chol_eps = float(chol_eps)
        self.kappa_floor = float(kappa_floor)
        self.kappa_max = float(kappa_max)
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        _an = str(activation).lower() if activation is not None else "tanh"
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[_an]
        emb_sum = sum(embedding_dims[k] for k in sorted(embedding_dims))
        layers: list[nn.Module] = []
        dim = input_dim + emb_sum
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        k = self.n_mixture
        # logits K + per comp: mu_va 2, chol 3, sin loc, cos loc, logkappa = 8K
        self.head = nn.Linear(dim, k + 8 * k)

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        embs = [self.embeddings[k](cat[k]) for k in sorted(cat.keys())]
        h = self.mlp(torch.cat([x_num] + embs, dim=-1))
        o = self.head(h)
        k = self.n_mixture
        logits = o[:, :k]
        rest = o[:, k:].view(-1, k, 8)
        mu_va = rest[:, :, :2]
        raw = rest[:, :, 2:5]
        l00 = F.softplus(raw[:, :, 0]) + self.chol_eps
        l10 = raw[:, :, 1]
        l11 = F.softplus(raw[:, :, 2]) + self.chol_eps
        z0 = torch.zeros(o.shape[0], k, device=o.device, dtype=o.dtype)
        row0 = torch.stack([l00, z0], dim=-1)
        row1 = torch.stack([l10, l11], dim=-1)
        L_va = torch.stack([row0, row1], dim=-2)

        s_loc = rest[:, :, 5]
        c_loc = rest[:, :, 6]
        loc_d = torch.atan2(s_loc, c_loc)
        kappa_d = F.softplus(rest[:, :, 7]) + self.kappa_floor

        return logits, mu_va, L_va, loc_d, kappa_d

    def joint_log_prob(
        self,
        y_std_full: torch.Tensor,
        d_raw_deg: torch.Tensor,
        logits: torch.Tensor,
        mu_va: torch.Tensor,
        L_va: torch.Tensor,
        loc_d: torch.Tensor,
        kappa_d: torch.Tensor,
    ) -> torch.Tensor:
        d_rad = torch.deg2rad(d_raw_deg)
        y_va = y_std_full[:, :2]
        return shared_mixture_bivariate_gaussian_vonmises_log_prob(
            y_va,
            d_rad,
            logits,
            mu_va,
            L_va,
            loc_d,
            kappa_d,
            kappa_max=self.kappa_max,
        )


def _build_chol_2x2(
    raw: torch.Tensor,
    *,
    chol_eps: float,
    diag_floor: float,
    diag_ceiling: float,
    use_diag_ceiling: bool,
    offdiag_tanh_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    raw: (B, K, 3) -> l00, l10, l11 per component.
    Returns (L_va, log_diag_scales, offdiag_l10) for diagnostics.
    """
    if use_diag_ceiling:
        span = float(diag_ceiling - diag_floor)
        l00 = diag_floor + span * torch.sigmoid(raw[:, :, 0])
        l11 = diag_floor + span * torch.sigmoid(raw[:, :, 2])
        log_diag = torch.log(l00.clamp_min(chol_eps))
    else:
        l00 = F.softplus(raw[:, :, 0]) + chol_eps
        l11 = F.softplus(raw[:, :, 2]) + chol_eps
        log_diag = torch.log(l00)
    if offdiag_tanh_scale and offdiag_tanh_scale > 0:
        l10 = float(offdiag_tanh_scale) * torch.tanh(raw[:, :, 1])
    else:
        l10 = raw[:, :, 1]
    z0 = torch.zeros(raw.shape[0], raw.shape[1], device=raw.device, dtype=raw.dtype)
    row0 = torch.stack([l00, z0], dim=-1)
    row1 = torch.stack([l10, l11], dim=-1)
    L_va = torch.stack([row0, row1], dim=-2)
    return L_va, log_diag, l10


class BoundedHierSharedMixtureGaussianVonMisesUNet(nn.Module):
    """
    Hierarchical shared mixture: global trunk + shrunken batter adapter;
    (v_ss, a) in bounded logit-zscore space; d_tilde via von Mises.

    batter_name is NOT in ordinary categorical embeddings — only player_embedding.
    """

    HEAD_FAMILY = "bounded_hier_shared_gaussian_vm_d"

    def __init__(
        self,
        input_dim: int,
        n_batters: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_mixture: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        player_embedding_dim: int = 8,
        player_effect_scale: float = 1.0,
        chol_eps: float = 1e-4,
        diag_floor: float = 0.05,
        diag_ceiling: float = 5.0,
        use_diag_ceiling: bool = True,
        kappa_floor: float = 1e-3,
        kappa_max: float = 120.0,
        ordinary_cat_cols: tuple[str, ...] = ("pitch_type", "stand", "p_throws"),
    ):
        super().__init__()
        if n_mixture < 1:
            raise ValueError("n_mixture must be >= 1")
        self.n_mixture = int(n_mixture)
        self.chol_eps = float(chol_eps)
        self.diag_floor = float(diag_floor)
        self.diag_ceiling = float(diag_ceiling)
        self.use_diag_ceiling = bool(use_diag_ceiling)
        self.kappa_floor = float(kappa_floor)
        self.kappa_max = float(kappa_max)
        self.player_effect_scale = float(player_effect_scale)
        self.ordinary_cat_cols = tuple(sorted(ordinary_cat_cols))

        self.player_embedding = nn.Embedding(n_batters, player_embedding_dim)
        nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.01)

        self.ordinary_embeddings = nn.ModuleDict(
            {
                k: nn.Embedding(vocab_sizes[k], embedding_dims[k])
                for k in self.ordinary_cat_cols
                if k in vocab_sizes
            }
        )
        _an = str(activation).lower() if activation is not None else "tanh"
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[_an]
        emb_sum = sum(embedding_dims[k] for k in self.ordinary_cat_cols if k in embedding_dims)
        layers: list[nn.Module] = []
        dim = input_dim + emb_sum
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.base_mlp = nn.Sequential(*layers)
        self.player_adapter = nn.Linear(player_embedding_dim, hidden_width, bias=False)
        nn.init.normal_(self.player_adapter.weight, mean=0.0, std=0.01)

        k = self.n_mixture
        self.head = nn.Linear(hidden_width, k + 8 * k)

    def _hidden(
        self,
        x_num: torch.Tensor,
        cat: dict[str, torch.Tensor],
        batter_id: torch.Tensor,
    ) -> torch.Tensor:
        embs = [
            self.ordinary_embeddings[k](cat[k])
            for k in self.ordinary_cat_cols
            if k in self.ordinary_embeddings
        ]
        h_in = torch.cat([x_num] + embs, dim=-1) if embs else x_num
        base_h = self.base_mlp(h_in)
        pe = self.player_embedding(batter_id)
        return base_h + self.player_effect_scale * self.player_adapter(pe)

    def forward(
        self,
        x_num: torch.Tensor,
        cat: dict[str, torch.Tensor],
        batter_id: torch.Tensor,
    ):
        h = self._hidden(x_num, cat, batter_id)
        o = self.head(h)
        k = self.n_mixture
        logits = o[:, :k]
        rest = o[:, k:].view(-1, k, 8)
        mu_va = rest[:, :, :2]
        raw_chol = rest[:, :, 2:5]
        L_va, log_diag, offdiag = _build_chol_2x2(
            raw_chol,
            chol_eps=self.chol_eps,
            diag_floor=self.diag_floor,
            diag_ceiling=self.diag_ceiling,
            use_diag_ceiling=self.use_diag_ceiling,
        )
        s_loc = rest[:, :, 5]
        c_loc = rest[:, :, 6]
        loc_d = torch.atan2(s_loc, c_loc)
        kappa_d = F.softplus(rest[:, :, 7]) + self.kappa_floor
        self._last_chol_diag = log_diag
        self._last_chol_offdiag = offdiag
        return logits, mu_va, L_va, loc_d, kappa_d

    def joint_log_prob(
        self,
        y_va_tilde: torch.Tensor,
        d_raw_deg: torch.Tensor,
        logits: torch.Tensor,
        mu_va: torch.Tensor,
        L_va: torch.Tensor,
        loc_d: torch.Tensor,
        kappa_d: torch.Tensor,
    ) -> torch.Tensor:
        d_rad = torch.deg2rad(d_raw_deg)
        y_va = y_va_tilde[:, :2] if y_va_tilde.shape[1] > 2 else y_va_tilde
        return shared_mixture_bivariate_gaussian_vonmises_log_prob(
            y_va,
            d_rad,
            logits,
            mu_va,
            L_va,
            loc_d,
            kappa_d,
            kappa_max=self.kappa_max,
        )

    @staticmethod
    def mixture_effective_components(pi: torch.Tensor) -> torch.Tensor:
        """K_eff = 1 / sum_k pi_k^2 per row."""
        return 1.0 / (pi.pow(2).sum(dim=-1).clamp_min(1e-12))

    @staticmethod
    def mixture_entropy(pi: torch.Tensor) -> torch.Tensor:
        p = pi.clamp_min(1e-12)
        return -(p * torch.log(p)).sum(dim=-1)


class BoundedPlayerSupportHierSharedMixtureGaussianVonMisesUNet(BoundedHierSharedMixtureGaussianVonMisesUNet):
    """
    Player-specific bounded logit targets + train-only circular d support at sample time;
    hierarchical player pooling with optional off-diagonal Cholesky tanh scaling.
    """

    HEAD_FAMILY = "bounded_player_support_hier_shared_gaussian_vm_d"

    def __init__(
        self,
        *args,
        offdiag_tanh_scale: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.offdiag_tanh_scale = float(offdiag_tanh_scale)

    def forward(
        self,
        x_num: torch.Tensor,
        cat: dict[str, torch.Tensor],
        batter_id: torch.Tensor,
    ):
        h = self._hidden(x_num, cat, batter_id)
        o = self.head(h)
        k = self.n_mixture
        logits = o[:, :k]
        rest = o[:, k:].view(-1, k, 8)
        mu_va = rest[:, :, :2]
        raw_chol = rest[:, :, 2:5]
        L_va, log_diag, offdiag = _build_chol_2x2(
            raw_chol,
            chol_eps=self.chol_eps,
            diag_floor=self.diag_floor,
            diag_ceiling=self.diag_ceiling,
            use_diag_ceiling=self.use_diag_ceiling,
            offdiag_tanh_scale=self.offdiag_tanh_scale,
        )
        s_loc = rest[:, :, 5]
        c_loc = rest[:, :, 6]
        loc_d = torch.atan2(s_loc, c_loc)
        kappa_d = F.softplus(rest[:, :, 7]) + self.kappa_floor
        self._last_chol_diag = log_diag
        self._last_chol_offdiag = offdiag
        return logits, mu_va, L_va, loc_d, kappa_d
