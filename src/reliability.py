"""Impact-threshold reliability of the MDN ensemble.

For thresholds d < {0.05, 0.02, 0.01} au, compares the predicted exceedance
probability P(log10 dmin < log10 d_thr) against the empirical clone fraction
on the test split, per (asteroid, window). Outputs a reliability diagram and
Brier scores (vs climatology) to results/.

Usage: python src/reliability.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import load_ensemble, ens_cdf

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

THRESHOLDS_AU = [0.05, 0.02, 0.01]

plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def main():
    d = np.load(ROOT / "data" / "dataset.npz", allow_pickle=True)
    X, Y = d["test_x"], d["test_y"]
    Ytr = d["train_y"]
    models, mu_x, sd_x, cols = load_ensemble(range(5))
    if cols is not None:
        X = X[:, cols]
    Xn = torch.tensor((X - mu_x) / sd_x, dtype=torch.float32)
    n_ast, _, n_win = Y.shape

    thr_log = torch.tensor([np.log10(t) for t in THRESHOLDS_AU],
                           dtype=torch.float32)
    p_pred = {t: [] for t in THRESHOLDS_AU}   # predicted prob
    p_emp = {t: [] for t in THRESHOLDS_AU}    # empirical clone fraction
    for w in range(n_win):
        wt = torch.full((n_ast,), w, dtype=torch.long)
        F = ens_cdf(models, Xn, wt, thr_log).numpy()   # (n_ast, 3)
        for i in range(n_ast):
            y = Y[i, :, w]
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            for k, t in enumerate(THRESHOLDS_AU):
                p_pred[t].append(F[i, k])
                p_emp[t].append(float((y < np.log10(t)).mean()))

    ytr = Ytr[np.isfinite(Ytr)]
    metrics = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, t in zip(axes, THRESHOLDS_AU):
        pp = np.array(p_pred[t])
        pe = np.array(p_emp[t])
        brier = float(np.mean((pp - pe) ** 2))
        p_clim = float((ytr < np.log10(t)).mean())
        brier_clim = float(np.mean((p_clim - pe) ** 2))
        metrics[f"d<{t}au"] = {
            "brier": brier, "brier_climatology": brier_clim,
            "base_rate": float(pe.mean()), "n": int(len(pp)),
        }
        # reliability curve: bin by predicted prob
        idx = np.digitize(pp, bins) - 1
        xs, ys, ns = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() < 5:
                continue
            xs.append(pp[m].mean())
            ys.append(pe[m].mean())
            ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, "o-", color="#4477aa")
        for xb, yb, nb in zip(xs, ys, ns):
            ax.annotate(str(nb), (xb, yb), fontsize=7,
                        xytext=(3, -9), textcoords="offset points")
        ax.set_title(f"$d<{t}$ au  (Brier {brier:.4f} vs clim "
                     f"{brier_clim:.4f})", fontsize=10)
        ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("empirical clone fraction")
    fig.suptitle("Impact-threshold reliability (test split)")
    fig.tight_layout()
    fig.savefig(FIGS / "reliability.png")
    (ROOT / "results" / "reliability.json").write_text(
        json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("saved results/figs/reliability.png")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
