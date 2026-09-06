"""Conditional full-covariance Gaussian mixture for stage-z (MDN via Cholesky L, Sigma = L L^T)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import VonMises


def _n_tril(d: int) -> int:
    return d * (d + 1) // 2


def tril_vec_to_chol(raw: torch.Tensor, d: int, chol_eps: float) -> torch.Tensor:
    """
    raw: (..., n_tril) -> lower Cholesky L (..., d, d) with positive diagonal (softplus + eps).
    """
    orig_shape = raw.shape[:-1]
    raw_m = raw.reshape(-1, _n_tril(d))
    n_batch = raw_m.shape[0]
    L = torch.zeros(n_batch, d, d, device=raw.device, dtype=raw.dtype)
    tril_i, tril_j = torch.tril_indices(d, d, device=raw.device)
    L[:, tril_i, tril_j] = raw_m
    diag = torch.arange(d, device=raw.device)
    L[:, diag, diag] = F.softplus(L[:, diag, diag]) + float(chol_eps)
    return L.view(*orig_shape, d, d)


def mdn_log_prob_chol(
    y: torch.Tensor,
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    *,
    log_eps: float = 1e-8,
) -> torch.Tensor:
    """
    Log p(y) for mixture of full-covariance Gaussians with Sigma_k = L_k L_k^T (L lower triangular).

    y: (B, D); mix_logits (B, K); mu (B, K, D); L (B, K, D, D).
    """
    B, D = y.shape
    K = mix_logits.shape[1]
    pi = F.softmax(mix_logits, dim=-1).clamp_min(log_eps)
    log_pi = torch.log(pi)

    diff = y.unsqueeze(1) - mu  # (B, K, D)
    Lb = L.reshape(B * K, D, D)
    db = diff.reshape(B * K, D, 1)
    z = torch.linalg.solve_triangular(Lb, db, upper=False).squeeze(-1)  # (B*K, D)
    quad = -0.5 * (z * z).sum(dim=-1).view(B, K)

    diag_L = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-8)
    sum_log_diag = torch.sum(torch.log(diag_L), dim=-1)  # (B, K); log|L| = sum log diag
    log_norm = -0.5 * D * math.log(2 * math.pi)
    log_comp = quad + log_norm - sum_log_diag

    return torch.logsumexp(log_pi + log_comp, dim=-1)


@torch.no_grad()
def sample_mdn_chol(
    mix_logits: torch.Tensor,
    mu: torch.Tensor,
    L: torch.Tensor,
    n_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample y ~ sum_k pi_k N(mu_k, L_k L_k^T) in the same space as training (standardized).

    mix_logits: (B, K); mu: (B, K, D); L: (B, K, D, D) lower Cholesky.
    Returns (B, n_samples, D).
    """
    B, K, D = mu.shape
    pi = F.softmax(mix_logits, dim=-1)
    comp = torch.multinomial(pi, n_samples, replacement=True, generator=generator)  # (B, S)
    b_idx = torch.arange(B, device=mu.device).unsqueeze(1).expand(B, n_samples)
    mu_sel = mu[b_idx, comp]
    L_sel = L[b_idx, comp]
    eps = torch.randn(B, n_samples, D, device=mu.device, dtype=mu.dtype, generator=generator)
    bmn = B * n_samples
    delta = torch.bmm(L_sel.reshape(bmn, D, D), eps.reshape(bmn, D, 1)).squeeze(-1)
    delta = delta.view(B, n_samples, D)
    return mu_sel + delta


def mixture_marginal_std_from_chol(L: torch.Tensor) -> torch.Tensor:
    """Sigma = L L^T; return sqrt(diag(Sigma)) with shape (B, K, D)."""
    # Sigma[b,k,i,i] = sum_m L[b,k,i,m]^2 for m <= i
    Sig = torch.matmul(L, L.transpose(-1, -2))
    v = torch.diagonal(Sig, dim1=-2, dim2=-1).clamp_min(1e-12)
    return torch.sqrt(v)


class ConditionalGaussianMixtureZ(nn.Module):
    """K-component full-covariance Gaussian mixture over z in R^D (standardized targets)."""

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        z_dim: int,
        n_components: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        chol_eps: float = 1e-5,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components >= 1")
        self.z_dim = int(z_dim)
        self.n_components = int(n_components)
        self.chol_eps = float(chol_eps)
        d = self.z_dim
        self._n_tril = _n_tril(d)

        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation.lower()]
        emb_sum = sum(embedding_dims[k] for k in sorted(embedding_dims))
        dim = input_dim + emb_sum
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        k = n_components
        out = k + k * d + k * self._n_tril
        self.head = nn.Linear(dim, out)

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        embs = [self.embeddings[k](cat[k]) for k in sorted(cat.keys())]
        h = self.mlp(torch.cat([x_num] + embs, dim=-1))
        o = self.head(h)
        k = self.n_components
        d = self.z_dim
        logits = o[:, :k]
        mu = o[:, k : k + k * d].view(-1, k, d)
        raw_tril = o[:, k + k * d :].view(-1, k, self._n_tril)
        L = tril_vec_to_chol(raw_tril, d, self.chol_eps)
        return logits, mu, L

    def log_prob(self, y: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return mdn_log_prob_chol(y, logits, mu, L)


def _angle_from_sincos(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Unconstrained s,c -> unit direction; mean angle in (-pi, pi]."""
    n = torch.sqrt(s * s + c * c).clamp_min(1e-12)
    return torch.atan2(s / n, c / n)


def mdn_hybrid_gauss_vm_log_prob(
    y_g: torch.Tensor,
    psi: torch.Tensor,
    theta: torch.Tensor,
    mix_logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    mu_theta: torch.Tensor,
    kappa_theta: torch.Tensor,
    *,
    log_eps: float = 1e-8,
) -> torch.Tensor:
    """
    Shared mixture weights; per k: N(m_g, L_g L_g^T) x VM(psi) x VM(theta).

    y_g: (B, 2) standardized x/e_y*; psi, theta: (B,) radians in [-pi, pi).
    mu_* : (B, K); kappa_* : (B, K) positive.
    """
    B, d_g = y_g.shape
    K = mix_logits.shape[1]
    pi = F.softmax(mix_logits, dim=-1).clamp_min(log_eps)
    log_pi = torch.log(pi)

    diff = y_g.unsqueeze(1) - mu_g  # (B, K, 2)
    Lb = L_g.reshape(B * K, d_g, d_g)
    db = diff.reshape(B * K, d_g, 1)
    z = torch.linalg.solve_triangular(Lb, db, upper=False).squeeze(-1)
    quad = -0.5 * (z * z).sum(dim=-1).view(B, K)
    diag_L = torch.diagonal(L_g, dim1=-2, dim2=-1).clamp_min(1e-8)
    sum_log_diag = torch.sum(torch.log(diag_L), dim=-1)
    log_norm_g = -0.5 * d_g * math.log(2 * math.pi)
    log_ng = quad + log_norm_g - sum_log_diag

    psi_b = psi.unsqueeze(-1).expand(-1, K)
    th_b = theta.unsqueeze(-1).expand(-1, K)
    dist_p = VonMises(mu_psi, kappa_psi)
    dist_t = VonMises(mu_theta, kappa_theta)
    log_vp = dist_p.log_prob(psi_b)
    log_vt = dist_t.log_prob(th_b)

    log_comp = log_ng + log_vp + log_vt
    return torch.logsumexp(log_pi + log_comp, dim=-1)


@torch.no_grad()
def sample_mdn_hybrid_gauss_vm(
    mix_logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    mu_theta: torch.Tensor,
    kappa_theta: torch.Tensor,
    n_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns y_g (B,S,2), psi (B,S), theta (B,S) in standardized / rad space.
    """
    B, K, d_g = mu_g.shape
    pi = F.softmax(mix_logits, dim=-1)
    comp = torch.multinomial(pi, n_samples, replacement=True, generator=generator)
    b_idx = torch.arange(B, device=mu_g.device).unsqueeze(1).expand(B, n_samples)
    mu_sel = mu_g[b_idx, comp]
    L_sel = L_g[b_idx, comp]
    eps = torch.randn(B, n_samples, d_g, device=mu_g.device, dtype=mu_g.dtype, generator=generator)
    bmn = B * n_samples
    delta = torch.bmm(L_sel.reshape(bmn, d_g, d_g), eps.reshape(bmn, d_g, 1)).squeeze(-1)
    y_g = (mu_sel + delta.view(B, n_samples, d_g)).clone()

    mpsi = mu_psi[b_idx, comp]
    kpsi = kappa_psi[b_idx, comp]
    mth = mu_theta[b_idx, comp]
    kth = kappa_theta[b_idx, comp]
    dist_p = VonMises(mpsi, kpsi)
    dist_t = VonMises(mth, kth)
    psi_s = dist_p.sample((1,)).squeeze(0)
    th_s = dist_t.sample((1,)).squeeze(0)
    return y_g, psi_s, th_s


@torch.no_grad()
def sample_mdn_gauss_psi_only(
    mix_logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    n_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 2D Gaussian block + von Mises(psi) only (no theta). Returns y_g (B,S,2), psi (B,S)."""
    B, K, d_g = mu_g.shape
    pi = F.softmax(mix_logits, dim=-1)
    comp = torch.multinomial(pi, n_samples, replacement=True, generator=generator)
    b_idx = torch.arange(B, device=mu_g.device).unsqueeze(1).expand(B, n_samples)
    mu_sel = mu_g[b_idx, comp]
    L_sel = L_g[b_idx, comp]
    eps = torch.randn(B, n_samples, d_g, device=mu_g.device, dtype=mu_g.dtype, generator=generator)
    bmn = B * n_samples
    delta = torch.bmm(L_sel.reshape(bmn, d_g, d_g), eps.reshape(bmn, d_g, 1)).squeeze(-1)
    y_g = (mu_sel + delta.view(B, n_samples, d_g)).clone()
    mpsi = mu_psi[b_idx, comp]
    kpsi = kappa_psi[b_idx, comp]
    dist_p = VonMises(mpsi, kpsi)
    psi_s = dist_p.sample((1,)).squeeze(0)
    return y_g, psi_s


class ConditionalHybridGaussVonMisesMixtureZ(nn.Module):
    """
    K-component mixture with shared pi_k(u): 2D full-cov Gaussian on standardized (x, e_y*)
    and independent von Mises factors for psi, theta (radians).
    """

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_components: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        chol_eps: float = 1e-5,
        kappa_floor: float = 1e-3,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components >= 1")
        self.n_components = int(n_components)
        self.d_g = 2
        self.chol_eps = float(chol_eps)
        self.kappa_floor = float(kappa_floor)
        self._n_tril = _n_tril(self.d_g)

        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation.lower()]
        emb_sum = sum(embedding_dims[k] for k in sorted(embedding_dims))
        dim = input_dim + emb_sum
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        k = n_components
        # logits K + mu_g 2K + chol 3K + psi(s,c) 2K + logk_psi K + theta(s,c) 2K + logk_th K = 12K
        out_dim = k * (1 + 2 + 3 + 2 + 1 + 2 + 1)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        embs = [self.embeddings[k](cat[k]) for k in sorted(cat.keys())]
        h = self.mlp(torch.cat([x_num] + embs, dim=-1))
        o = self.head(h)
        k = self.n_components
        logits = o[:, :k]
        mu_g = o[:, k : k + 2 * k].view(-1, k, 2)
        raw_tril = o[:, k + 2 * k : k + 5 * k].view(-1, k, self._n_tril)
        L_g = tril_vec_to_chol(raw_tril, self.d_g, self.chol_eps)
        o2 = o[:, k + 5 * k :]
        s_psi = o2[:, : 2 * k].view(-1, k, 2)[..., 0]
        c_psi = o2[:, : 2 * k].view(-1, k, 2)[..., 1]
        mu_psi = _angle_from_sincos(s_psi, c_psi)
        logk_psi = o2[:, 2 * k : 3 * k]
        kappa_psi = F.softplus(logk_psi) + self.kappa_floor

        s_th = o2[:, 3 * k : 5 * k].view(-1, k, 2)[..., 0]
        c_th = o2[:, 3 * k : 5 * k].view(-1, k, 2)[..., 1]
        mu_theta = _angle_from_sincos(s_th, c_th)
        logk_th = o2[:, 5 * k : 6 * k]
        kappa_theta = F.softplus(logk_th) + self.kappa_floor

        return logits, mu_g, L_g, mu_psi, kappa_psi, mu_theta, kappa_theta

    def log_prob(
        self,
        y_g: torch.Tensor,
        psi: torch.Tensor,
        theta: torch.Tensor,
        logits: torch.Tensor,
        mu_g: torch.Tensor,
        L_g: torch.Tensor,
        mu_psi: torch.Tensor,
        kappa_psi: torch.Tensor,
        mu_theta: torch.Tensor,
        kappa_theta: torch.Tensor,
    ) -> torch.Tensor:
        return mdn_hybrid_gauss_vm_log_prob(
            y_g,
            psi,
            theta,
            logits,
            mu_g,
            L_g,
            mu_psi,
            kappa_psi,
            mu_theta,
            kappa_theta,
        )


def mdn_hybrid_gauss_vm_psi_only_log_prob(
    y_g: torch.Tensor,
    psi: torch.Tensor,
    mix_logits: torch.Tensor,
    mu_g: torch.Tensor,
    L_g: torch.Tensor,
    mu_psi: torch.Tensor,
    kappa_psi: torch.Tensor,
    *,
    log_eps: float = 1e-8,
) -> torch.Tensor:
    """Mixture log p for 2D Gaussian (x,e_y*) x von Mises(psi) only (no theta factor)."""
    B, d_g = y_g.shape
    K = mix_logits.shape[1]
    pi = F.softmax(mix_logits, dim=-1).clamp_min(log_eps)
    log_pi = torch.log(pi)

    diff = y_g.unsqueeze(1) - mu_g
    Lb = L_g.reshape(B * K, d_g, d_g)
    db = diff.reshape(B * K, d_g, 1)
    z = torch.linalg.solve_triangular(Lb, db, upper=False).squeeze(-1)
    quad = -0.5 * (z * z).sum(dim=-1).view(B, K)
    diag_L = torch.diagonal(L_g, dim1=-2, dim2=-1).clamp_min(1e-8)
    sum_log_diag = torch.sum(torch.log(diag_L), dim=-1)
    log_norm_g = -0.5 * d_g * math.log(2 * math.pi)
    log_ng = quad + log_norm_g - sum_log_diag

    psi_b = psi.unsqueeze(-1).expand(-1, K)
    dist_p = VonMises(mu_psi, kappa_psi)
    log_vp = dist_p.log_prob(psi_b)

    log_comp = log_ng + log_vp
    return torch.logsumexp(log_pi + log_comp, dim=-1)


def trunc_normal_01_log_prob(ex: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log pdf of Normal(mu,sigma) truncated to [0, 0.6] (broadcast-safe)."""
    sigma = sigma.clamp_min(1e-4)
    lo = ex.new_tensor(0.0)
    hi = ex.new_tensor(0.6)
    dist = torch.distributions.Normal(mu, sigma)
    zlo = (lo - mu) / sigma
    zhi = (hi - mu) / sigma
    logZ = torch.log(dist.cdf(zhi) - dist.cdf(zlo) + 1e-18)
    return dist.log_prob(ex) - logZ


class ConditionalHybridGaussVonMisesTruncExZ(nn.Module):
    """
    K-way mixture: full-cov 2D Gaussian on standardized (x, e_y*), von Mises(psi),
    plus a **single** truncated-normal head for e_x | (x, e_y*, psi, context).
    """

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_components: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        chol_eps: float = 1e-5,
        kappa_floor: float = 1e-3,
        sigma_floor: float = 1e-3,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components >= 1")
        self.n_components = int(n_components)
        self.d_g = 2
        self.chol_eps = float(chol_eps)
        self.kappa_floor = float(kappa_floor)
        self.sigma_floor = float(sigma_floor)
        self._n_tril = _n_tril(self.d_g)

        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation.lower()]
        emb_sum = sum(embedding_dims[k] for k in sorted(vocab_sizes.keys()))
        dim = input_dim + emb_sum
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        self.hidden_width = int(hidden_width)
        k = n_components
        out_dim = k * (1 + 2 + 3 + 2 + 1)
        self.head = nn.Linear(self.hidden_width, out_dim)
        # e_x head: context h + standardized (x,e_y*) from labels + sin/cos(psi)
        self.ex_head = nn.Sequential(
            nn.Linear(self.hidden_width + 4, self.hidden_width),
            nn.Tanh(),
            nn.Linear(self.hidden_width, 2),
        )

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        embs = [self.embeddings[k](cat[k]) for k in sorted(self.embeddings.keys())]
        h = self.mlp(torch.cat([x_num] + embs, dim=-1))
        o = self.head(h)
        k = self.n_components
        logits = o[:, :k]
        mu_g = o[:, k : k + 2 * k].view(-1, k, 2)
        raw_tril = o[:, k + 2 * k : k + 5 * k].view(-1, k, self._n_tril)
        L_g = tril_vec_to_chol(raw_tril, self.d_g, self.chol_eps)
        o2 = o[:, k + 5 * k :]
        s_psi = o2[:, : 2 * k].view(-1, k, 2)[..., 0]
        c_psi = o2[:, : 2 * k].view(-1, k, 2)[..., 1]
        mu_psi = _angle_from_sincos(s_psi, c_psi)
        logk_psi = o2[:, 2 * k : 3 * k]
        kappa_psi = F.softplus(logk_psi) + self.kappa_floor
        return logits, mu_g, L_g, mu_psi, kappa_psi, h

    def ex_params(self, h: torch.Tensor, y_g: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditional truncated-normal parameters (B,). ``y_g`` is standardized (x,e_y*)."""
        if psi.dim() == 1:
            psi = psi.unsqueeze(-1)
        sp = torch.sin(psi)
        cp = torch.cos(psi)
        ex_in = torch.cat([h, y_g.detach(), sp.detach(), cp.detach()], dim=-1)
        oex = self.ex_head(ex_in)
        mu_ex = oex[:, 0]
        sigma_ex = F.softplus(oex[:, 1]) + self.sigma_floor
        return mu_ex, sigma_ex

    def ex_params_batched(
        self, h: torch.Tensor, y_g: torch.Tensor, psi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h`` (B,H), ``y_g`` (B,S,2), ``psi`` (B,S) -> (B*S,) mu and sigma."""
        b, s, _ = y_g.shape
        hh = h.unsqueeze(1).expand(-1, s, -1).reshape(b * s, -1)
        yf = y_g.reshape(b * s, 2)
        pf = psi.reshape(b * s)
        return self.ex_params(hh, yf, pf)

    def log_prob_joint(
        self,
        y_g: torch.Tensor,
        psi: torch.Tensor,
        ex: torch.Tensor,
        logits: torch.Tensor,
        mu_g: torch.Tensor,
        L_g: torch.Tensor,
        mu_psi: torch.Tensor,
        kappa_psi: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        lp_g = mdn_hybrid_gauss_vm_psi_only_log_prob(y_g, psi, logits, mu_g, L_g, mu_psi, kappa_psi)
        mu_ex, sig_ex = self.ex_params(h, y_g, psi)
        lp_x = trunc_normal_01_log_prob(ex, mu_ex, sig_ex)
        return lp_g + lp_x


def tril_vec_to_chol_bounded(
    raw: torch.Tensor,
    d: int,
    *,
    chol_eps: float,
    diag_floor: float,
    diag_ceiling: float,
    use_diag_ceiling: bool,
    offdiag_tanh_scale: float,
) -> torch.Tensor:
    """Lower Cholesky with bounded diagonal and tanh-bounded off-diagonal."""
    orig_shape = raw.shape[:-1]
    raw_m = raw.reshape(-1, _n_tril(d))
    n_batch = raw_m.shape[0]
    L = torch.zeros(n_batch, d, d, device=raw.device, dtype=raw.dtype)
    tril_i, tril_j = torch.tril_indices(d, d, device=raw.device)
    L[:, tril_i, tril_j] = raw_m
    diag_idx = torch.arange(d, device=raw.device)
    raw_diag = L[:, diag_idx, diag_idx]
    if use_diag_ceiling:
        lo = float(diag_floor)
        hi = float(diag_ceiling)
        L[:, diag_idx, diag_idx] = lo + (hi - lo) * torch.sigmoid(raw_diag)
    else:
        L[:, diag_idx, diag_idx] = F.softplus(raw_diag) + float(chol_eps)
    off = tril_i != tril_j
    if off.any():
        oi, oj = tril_i[off], tril_j[off]
        L[:, oi, oj] = float(offdiag_tanh_scale) * torch.tanh(L[:, oi, oj])
    return L.view(*orig_shape, d, d)


def _gaussian_log_prob_2d(y: torch.Tensor, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """y (B,2); mu (B,K,2); L (B,K,2,2) -> log N (B,K)."""
    B, K, d = mu.shape
    diff = y.unsqueeze(1) - mu
    Lb = L.reshape(B * K, d, d)
    db = diff.reshape(B * K, d, 1)
    z = torch.linalg.solve_triangular(Lb, db, upper=False).squeeze(-1)
    quad = -0.5 * (z * z).sum(dim=-1).view(B, K)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-8)
    sum_log_diag = torch.sum(torch.log(diag_L), dim=-1)
    log_norm = -0.5 * d * math.log(2 * math.pi)
    return quad + log_norm - sum_log_diag


def _vonmises_log_prob_1d(psi: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """psi (B,); mu,kappa (B,K) -> (B,K)."""
    psi_b = psi.unsqueeze(-1).expand_as(mu)
    return VonMises(mu, kappa).log_prob(psi_b)


def _normal_log_prob_1d(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """y (B,); mu,sigma (B,K) -> (B,K)."""
    y_b = y.unsqueeze(-1).expand_as(mu)
    return torch.distributions.Normal(mu, sigma.clamp_min(1e-4)).log_prob(y_b)


class BranchAutoregBoundedHybridMDNZ(nn.Module):
    """
    branch_autoreg_bounded_hybrid_mdn_z: shared pi_b(c), Gaussian on (r_x_std, r_y_std),
    branch-specific VM(psi | r_x, r_y, b, c) and Normal(r_ex_std | r_x, r_y, psi, b, c).
    """

    def __init__(
        self,
        input_dim: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_components: int,
        hidden_width: int = 64,
        n_hidden: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        branch_embedding_dim: int = 8,
        *,
        chol_eps: float = 1e-4,
        diag_floor: float = 0.05,
        diag_ceiling: float = 5.0,
        use_diag_ceiling: bool = True,
        offdiag_tanh_scale: float = 2.0,
        kappa_floor: float = 1e-3,
        kappa_max: float = 120.0,
        sigma_floor: float = 0.05,
        sigma_ceiling: float = 5.0,
        use_sigma_ceiling: bool = True,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components >= 1")
        self.n_components = int(n_components)
        self.d_xy = 2
        self.branch_embedding_dim = int(branch_embedding_dim)
        self.chol_eps = float(chol_eps)
        self.diag_floor = float(diag_floor)
        self.diag_ceiling = float(diag_ceiling)
        self.use_diag_ceiling = bool(use_diag_ceiling)
        self.offdiag_tanh_scale = float(offdiag_tanh_scale)
        self.kappa_floor = float(kappa_floor)
        self.kappa_max = float(kappa_max)
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceiling = float(sigma_ceiling)
        self.use_sigma_ceiling = bool(use_sigma_ceiling)
        self._n_tril_xy = _n_tril(self.d_xy)

        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(vocab_sizes[k], embedding_dims[k]) for k in vocab_sizes}
        )
        self.branch_embeddings = nn.Embedding(n_components, branch_embedding_dim)
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation.lower()]
        emb_sum = sum(embedding_dims[k] for k in sorted(embedding_dims))
        dim = input_dim + emb_sum
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_width
        self.mlp = nn.Sequential(*layers)
        self.hidden_width = int(hidden_width)
        k = n_components
        out_dim = k * (1 + 2 + self._n_tril_xy)
        self.mixture_head = nn.Linear(self.hidden_width, out_dim)
        psi_in = self.hidden_width + branch_embedding_dim + 2
        self.psi_head = nn.Sequential(
            nn.Linear(psi_in, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 3),
        )
        ex_in = psi_in + 2
        self.ex_head = nn.Sequential(
            nn.Linear(ex_in, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 2),
        )

    def trunk(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]) -> torch.Tensor:
        embs = [self.embeddings[k](cat[k]) for k in sorted(self.embeddings.keys())]
        return self.mlp(torch.cat([x_num] + embs, dim=-1))

    def mixture_params(self, h: torch.Tensor):
        o = self.mixture_head(h)
        k = self.n_components
        logits = o[:, :k]
        mu_xy = o[:, k : k + 2 * k].view(-1, k, 2)
        raw_tril = o[:, k + 2 * k :].view(-1, k, self._n_tril_xy)
        L_xy = tril_vec_to_chol_bounded(
            raw_tril,
            self.d_xy,
            chol_eps=self.chol_eps,
            diag_floor=self.diag_floor,
            diag_ceiling=self.diag_ceiling,
            use_diag_ceiling=self.use_diag_ceiling,
            offdiag_tanh_scale=self.offdiag_tanh_scale,
        )
        return logits, mu_xy, L_xy

    def _branch_expand(self, h: torch.Tensor, r_xy: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        K = self.n_components
        h_exp = h.unsqueeze(1).expand(-1, K, -1)
        be = self.branch_embeddings.weight.unsqueeze(0).expand(B, -1, -1)
        r_exp = r_xy.unsqueeze(1).expand(-1, K, -1)
        return torch.cat([h_exp, be, r_exp], dim=-1)

    def psi_params_all_branches(
        self, h: torch.Tensor, r_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mu_psi, kappa: (B, K). r_xy detached for label conditioning."""
        inp = self._branch_expand(h, r_xy.detach())
        B, K, _ = inp.shape
        out = self.psi_head(inp.reshape(B * K, -1)).reshape(B, K, 3)
        mu = _angle_from_sincos(out[..., 0], out[..., 1])
        kappa = F.softplus(out[..., 2]) + self.kappa_floor
        return mu, torch.clamp(kappa, max=self.kappa_max)

    def ex_params_all_branches(
        self, h: torch.Tensor, r_xy: torch.Tensor, psi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mu_ex, sigma_ex: (B, K)."""
        K = self.n_components
        sp = torch.sin(psi.detach()).unsqueeze(-1).unsqueeze(-1).expand(-1, K, -1)
        cp = torch.cos(psi.detach()).unsqueeze(-1).unsqueeze(-1).expand(-1, K, -1)
        inp = torch.cat([self._branch_expand(h, r_xy.detach()), sp, cp], dim=-1)
        B, K, _ = inp.shape
        out = self.ex_head(inp.reshape(B * K, -1)).reshape(B, K, 2)
        mu = out[..., 0]
        if self.use_sigma_ceiling:
            lo, hi = self.sigma_floor, self.sigma_ceiling
            sigma = lo + (hi - lo) * torch.sigmoid(out[..., 1])
        else:
            sigma = F.softplus(out[..., 1]) + self.sigma_floor
        return mu, sigma

    def forward(self, x_num: torch.Tensor, cat: dict[str, torch.Tensor]):
        h = self.trunk(x_num, cat)
        return (*self.mixture_params(h), h)

    def log_prob_components(
        self,
        r_xy: torch.Tensor,
        psi: torch.Tensor,
        r_ex: torch.Tensor,
        logits: torch.Tensor,
        mu_xy: torch.Tensor,
        L_xy: torch.Tensor,
        h: torch.Tensor,
        *,
        T_x: float = 1.0,
        T_y: float = 1.0,
        T_psi: float = 1.0,
        T_ex: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns log p (B,) and factor logs (B,) for xy, psi, ex (mixture-marginalized)."""
        from .calibration_z import _hybrid_scaled_L

        tx = L_xy.new_tensor(float(T_x), device=L_xy.device, dtype=L_xy.dtype)
        ty = L_xy.new_tensor(float(T_y), device=L_xy.device, dtype=L_xy.dtype)
        Lcal = _hybrid_scaled_L(L_xy, tx, ty)
        mu_psi, kappa = self.psi_params_all_branches(h, r_xy)
        kappa = kappa / float(T_psi)
        mu_ex, sigma = self.ex_params_all_branches(h, r_xy, psi)
        sigma = sigma * math.sqrt(float(T_ex))

        log_pi = F.log_softmax(logits, dim=-1)
        log_g = _gaussian_log_prob_2d(r_xy, mu_xy, Lcal)
        log_vm = _vonmises_log_prob_1d(psi, mu_psi, kappa)
        log_ex = _normal_log_prob_1d(r_ex, mu_ex, sigma)
        log_branch = log_pi + log_g + log_vm + log_ex
        log_joint = torch.logsumexp(log_branch, dim=-1)
        log_xy = torch.logsumexp(log_pi + log_g, dim=-1)
        log_gvm = torch.logsumexp(log_pi + log_g + log_vm, dim=-1)
        log_psi_inc = log_gvm - log_xy
        log_ex_inc = log_joint - log_gvm
        return log_joint, {"xy": log_xy, "psi": log_psi_inc, "ex": log_ex_inc}

    def log_prob_model_space(
        self,
        r_xy: torch.Tensor,
        psi: torch.Tensor,
        r_ex: torch.Tensor,
        logits: torch.Tensor,
        mu_xy: torch.Tensor,
        L_xy: torch.Tensor,
        h: torch.Tensor,
        **temp_kw,
    ) -> torch.Tensor:
        lp, _ = self.log_prob_components(
            r_xy, psi, r_ex, logits, mu_xy, L_xy, h, **temp_kw
        )
        return lp

    def regularization_terms(self, logits: torch.Tensor, L_xy: torch.Tensor) -> dict[str, torch.Tensor]:
        pi = F.softmax(logits, dim=-1)
        ent = -(pi * torch.log(pi.clamp_min(1e-8))).sum(dim=-1).mean()
        diag = torch.diagonal(L_xy, dim1=-2, dim2=-1)
        off = L_xy.clone()
        d = self.d_xy
        for i in range(d):
            off[..., i, i] = 0.0
        return {
            "mixture_entropy": ent,
            "offdiag_l2": (off**2).mean(),
            "log_diag_l2": (torch.log(diag.clamp_min(1e-8)) ** 2).mean(),
            "branch_emb_l2": (self.branch_embeddings.weight**2).mean(),
        }


@torch.no_grad()
def sample_branch_autoreg_bounded_mdn(
    model: BranchAutoregBoundedHybridMDNZ,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    *,
    n_samples: int,
    T_x: float = 1.0,
    T_y: float = 1.0,
    T_psi: float = 1.0,
    T_ex: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (r_xy_std, psi_rad, r_ex_std) with shape (B, S, *).
    """
    from .calibration_z import _hybrid_scaled_L

    h = model.trunk(x_num, cat)
    logits, mu_xy, L_xy = model.mixture_params(h)
    B, K, _ = mu_xy.shape
    pi = F.softmax(logits, dim=-1)
    comp = torch.multinomial(pi, n_samples, replacement=True, generator=generator)
    b_idx = torch.arange(B, device=x_num.device).unsqueeze(1).expand(B, n_samples)

    tx = L_xy.new_tensor(float(T_x))
    ty = L_xy.new_tensor(float(T_y))
    Lcal = _hybrid_scaled_L(L_xy, tx, ty)
    mu_sel = mu_xy[b_idx, comp]
    L_sel = Lcal[b_idx, comp]
    eps = torch.randn(B, n_samples, 2, device=x_num.device, dtype=x_num.dtype, generator=generator)
    bmn = B * n_samples
    delta = torch.bmm(L_sel.reshape(bmn, 2, 2), eps.reshape(bmn, 2, 1)).squeeze(-1)
    r_xy = (mu_sel + delta.view(B, n_samples, 2)).clone()

    r_flat = r_xy.reshape(bmn, 2)
    hf = h.unsqueeze(1).expand(-1, n_samples, -1).reshape(bmn, -1)
    mu_psi, kappa = model.psi_params_all_branches(hf, r_flat)
    # use selected branch only
    k_sel = comp.reshape(bmn)
    mu_p = mu_psi[torch.arange(bmn, device=x_num.device), k_sel]
    kap_p = (kappa[torch.arange(bmn, device=x_num.device), k_sel] / float(T_psi)).clamp_min(1e-3)
    psi_s = VonMises(mu_p, kap_p).sample((1,)).squeeze(0).view(B, n_samples)

    sp = torch.sin(psi_s).reshape(bmn)
    cp = torch.cos(psi_s).reshape(bmn)
    mu_ex, sig_ex = model.ex_params_all_branches(hf, r_flat, psi_s.reshape(bmn))
    mu_e = mu_ex[torch.arange(bmn, device=x_num.device), k_sel]
    sig_e = (sig_ex[torch.arange(bmn, device=x_num.device), k_sel] * math.sqrt(float(T_ex))).clamp_min(1e-4)
    r_ex = torch.distributions.Normal(mu_e, sig_e).sample((1,)).squeeze(0).view(B, n_samples)
    return r_xy, psi_s, r_ex

