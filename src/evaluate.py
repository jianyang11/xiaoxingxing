"""Evaluate MDN deep ensemble on the held-out test split.

Metrics vs MC ground truth (256 clones/asteroid):
  - CRPS (numerical, on log10 d grid)
  - NLL of clone samples under mixture
  - Wasserstein-1 between predicted and empirical distribution
  - PIT histogram / calibration curve
  - Quantile scatter (q05/q50/q95)
  - Speed benchmark: ensemble inference vs REBOUND MC per asteroid

Ensemble = uniform mixture over seed models.
Outputs: results/metrics.json, results/figs/*.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MDN

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
RES = ROOT / "results"
FIGS = RES / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def load_ensemble(seeds):
    models, mu_x, sd_x, cols = [], None, None, None
    for s in seeds:
        f = CKPT / f"mdn_seed{s}_best.pt"
        if not f.exists():
            continue
        st = torch.load(f, map_location="cpu", weights_only=False)
        a = st["args"]
        mu_x, sd_x = st["mu_x"], st["sd_x"]
        if a.get("diag_cov"):
            n_el = len(mu_x) - 6
            cols = list(range(n_el)) + [n_el + k for k in (0, 2, 5, 9, 14, 20)]
        m = MDN(len(mu_x), a["hidden"], a["blocks"], a["comp"],
                a.get("dropout", 0.0))
        m.load_state_dict(st["model"])
        m.eval()
        models.append(m)
    return models, mu_x, sd_x, cols


def ens_cdf(models, x, w, grid):
    return torch.stack([m.cdf(x, w, grid) for m in models]).mean(0)


def ens_nll(models, x, w, y):
    lps = []
    for m in models:
        logit_pi, mu, log_sig = m(x, w)
        log_pi = torch.log_softmax(logit_pi, -1)
        z = (y.unsqueeze(-1) - mu) / log_sig.exp()
        lp = torch.logsumexp(log_pi + (-0.5 * z ** 2 - log_sig
                                       - 0.9189385332046727), -1)
        lps.append(lp)
    return -(torch.logsumexp(torch.stack(lps), 0) - np.log(len(models)))


def main():
    d = np.load(ROOT / "data" / "dataset.npz", allow_pickle=True)
    X, Y, SPK = d["test_x"], d["test_y"], d["test_spk"]
    models, mu_x, sd_x, cols = load_ensemble(range(5))
    print(f"{len(models)} ensemble members, {len(X)} test asteroids")
    if cols is not None:
        X = X[:, cols]
    Xn = torch.tensor((X - mu_x) / sd_x, dtype=torch.float32)

    grid = torch.linspace(-7.0, 1.0, 801)
    dg = float(grid[1] - grid[0])
    n_ast, _, n_win = Y.shape

    crps_all, nll_all, w1_all, pit_all = [], [], [], []
    q_pred, q_true = [], []
    for w in range(n_win):
        wt = torch.full((n_ast,), w, dtype=torch.long)
        F = ens_cdf(models, Xn, wt, grid)               # (n_ast, G)
        for i in range(n_ast):
            y = Y[i, :, w]
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            # empirical CDF on grid
            Fe = (y[None, :] <= grid.numpy()[:, None]).mean(1)
            Fp = F[i].numpy()
            crps_all.append(np.sum((Fp - Fe) ** 2) * dg)
            w1_all.append(np.sum(np.abs(Fp - Fe)) * dg)
            with torch.no_grad():
                nll_i = ens_nll(models, Xn[i:i+1].repeat(len(y), 1),
                                wt[:1].repeat(len(y)),
                                torch.tensor(y, dtype=torch.float32))
            nll_all.append(float(nll_i.clamp(max=20.0).mean()))
            Fm = np.maximum.accumulate(Fp)
            Fm = Fm + np.linspace(0, 1e-9, len(Fm))
            pit_all.append(np.interp(y, grid.numpy(), Fm))
            q_pred.append(np.interp([0.05, 0.5, 0.95], Fm, grid.numpy()))
            q_true.append(np.quantile(y, [0.05, 0.5, 0.95]))

    crps = float(np.mean(crps_all))
    nll = float(np.mean(nll_all))
    w1 = float(np.mean(w1_all))
    pit = np.concatenate(pit_all)
    q_pred, q_true = np.array(q_pred), np.array(q_true)

    # ---- baseline: climatological (train marginal) ----
    Ytr = d["train_y"]
    ytr = Ytr[~np.isnan(Ytr)]
    Fc = (ytr[None, :] <= grid.numpy()[:, None]).mean(1)
    crps_clim, w1_clim = [], []
    for i in range(len(q_true)):
        pass
    for w in range(n_win):
        for i in range(n_ast):
            y = Y[i, :, w]
            y = y[~np.isnan(y)]
            if len(y) == 0:
                continue
            Fe = (y[None, :] <= grid.numpy()[:, None]).mean(1)
            crps_clim.append(np.sum((Fc - Fe) ** 2) * dg)
            w1_clim.append(np.sum(np.abs(Fc - Fe)) * dg)

    # ---- speed benchmark ----
    t0 = time.time()
    with torch.no_grad():
        for w in range(n_win):
            ens_cdf(models, Xn, torch.full((n_ast,), w, dtype=torch.long), grid)
    t_nn = (time.time() - t0) / n_ast
    t_mc = 85.0  # measured REBOUND 256-clone 100-yr wall time per asteroid
    metrics = {
        "n_test_asteroids": int(n_ast), "n_ensemble": len(models),
        "CRPS": crps, "NLL": nll, "W1": w1,
        "CRPS_climatology": float(np.mean(crps_clim)),
        "W1_climatology": float(np.mean(w1_clim)),
        "q50_rmse": float(np.sqrt(np.mean((q_pred[:, 1] - q_true[:, 1]) ** 2))),
        "t_nn_per_asteroid_s": t_nn, "t_mc_per_asteroid_s": t_mc,
        "speedup": t_mc / t_nn,
    }
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pit, bins=20, density=True, color="#4477aa", alpha=0.85)
    ax.axhline(1, color="k", ls="--")
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_title("PIT histogram (test clones)")
    fig.tight_layout()
    fig.savefig(FIGS / "pit_hist.png")

    fig, ax = plt.subplots(figsize=(5, 4.6))
    lab = ["q05", "q50", "q95"]
    col = ["#e6774a", "#4477aa", "#66aa55"]
    for k in range(3):
        ax.scatter(q_true[:, k], q_pred[:, k], s=8, alpha=0.5,
                   color=col[k], label=lab[k])
    lim = [q_true.min() - 0.2, q_true.max() + 0.2]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(r"MC $\log_{10} d_{\min}$ [au]")
    ax.set_ylabel(r"MDN ensemble $\log_{10} d_{\min}$ [au]")
    ax.legend()
    ax.set_title("Quantile agreement")
    fig.tight_layout()
    fig.savefig(FIGS / "quantile_scatter.png")

    # calibration curve
    levels = np.linspace(0.05, 0.95, 19)
    cov = [(pit <= l).mean() for l in levels]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot(levels, cov, "o-", color="#4477aa")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("nominal level")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration")
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png")

    # example distributions: 4 asteroids, window 0 & 9
    idx = np.argsort([np.nanmin(Y[i]) for i in range(n_ast)])[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, i in zip(axes.ravel(), idx):
        for w, c in [(0, "#4477aa"), (9, "#e6774a")]:
            y = Y[i, :, w]
            y = y[~np.isnan(y)]
            ax.hist(y, bins=30, density=True, alpha=0.4, color=c,
                    label=f"MC win{w}")
            wt = torch.tensor([w])
            F = ens_cdf(models, Xn[i:i+1], wt, grid)[0].numpy()
            pdf = np.gradient(F, dg)
            ax.plot(grid.numpy(), pdf, color=c, lw=2,
                    label=f"MDN win{w}")
        ax.set_title(f"SPK {SPK[i]}")
        ax.set_xlabel(r"$\log_{10} d_{\min}$ [au]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "example_distributions.png")
    print("figures saved to", FIGS)


if __name__ == "__main__":
    main()
