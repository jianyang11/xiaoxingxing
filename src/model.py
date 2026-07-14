"""Mixture Density Network for p(log10 d_min | orbit, covariance, window)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_WINDOWS = 10


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return F.silu(x + self.net(x))


class MDN(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4, n_comp=8):
        super().__init__()
        self.n_comp = n_comp
        self.win_emb = nn.Embedding(N_WINDOWS, 16)
        self.inp = nn.Linear(in_dim + 16, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 3 * n_comp)

    def forward(self, x, w):
        h = torch.cat([x, self.win_emb(w)], dim=-1)
        h = self.blocks(F.silu(self.inp(h)))
        out = self.head(h)
        logit_pi, mu, log_sig = out.chunk(3, dim=-1)
        log_sig = log_sig.clamp(-5.0, 1.5)
        return logit_pi, mu, log_sig

    def nll(self, x, w, y):
        """y: (B,) log10 dmin"""
        logit_pi, mu, log_sig = self.forward(x, w)
        log_pi = F.log_softmax(logit_pi, dim=-1)
        z = (y.unsqueeze(-1) - mu) / log_sig.exp()
        log_prob = -0.5 * z ** 2 - log_sig - 0.9189385332046727
        return -(torch.logsumexp(log_pi + log_prob, dim=-1)).mean()

    @torch.no_grad()
    def cdf(self, x, w, y_grid):
        """CDF at y_grid (G,) for each row -> (B, G)"""
        logit_pi, mu, log_sig = self.forward(x, w)
        pi = F.softmax(logit_pi, dim=-1)
        z = (y_grid.view(1, -1, 1) - mu.unsqueeze(1)) / log_sig.exp().unsqueeze(1)
        comp_cdf = 0.5 * (1 + torch.erf(z / 2 ** 0.5))
        return (pi.unsqueeze(1) * comp_cdf).sum(-1)

    @torch.no_grad()
    def sample(self, x, w, n):
        logit_pi, mu, log_sig = self.forward(x, w)
        pi = F.softmax(logit_pi, dim=-1)
        comp = torch.multinomial(pi, n, replacement=True)          # (B, n)
        mu_s = torch.gather(mu, 1, comp)
        sig_s = torch.gather(log_sig, 1, comp).exp()
        return mu_s + sig_s * torch.randn_like(mu_s)
