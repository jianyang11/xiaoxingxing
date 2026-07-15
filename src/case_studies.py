"""Case studies on well-known PHAs (Apophis, Bennu).

Generates high-fidelity MC ground truth (256 clones, 1.8 d sampling) for each
case-study target, compares against the MDN ensemble prediction per decadal
window, and produces per-target figures + a JSON summary.

These targets are excluded from the training split (see dataset.py
CASE_STUDY_EXCLUDE) so the comparison is out-of-sample.

Usage: python src/case_studies.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import features_from_npz
from evaluate import load_ensemble, ens_cdf

ROOT = Path(__file__).resolve().parent.parent
COVDIR = ROOT / "data" / "raw" / "cov"
CASEDIR = ROOT / "data" / "sim" / "case"
FIGS = ROOT / "results" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CASES = {"20099942": "Apophis", "20101955": "Bennu"}
N_CLONES = 256

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def ensure_mc(spkid):
    out = CASEDIR / f"{spkid}.npz"
    if out.exists():
        return out
    from propagate import run_one
    CASEDIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp.npz")
    ok = run_one(COVDIR / f"{spkid}.json", N_CLONES, int(spkid) % (2 ** 31),
                 tmp, sample_days=1.8)
    if not ok:
        raise RuntimeError(f"MC failed for {spkid}")
    tmp.replace(out)
    return out


def main():
    models, mu_x, sd_x, cols = load_ensemble(range(5))
    print(f"{len(models)} ensemble members")
    grid = torch.linspace(-7.0, 1.0, 801)
    dg = float(grid[1] - grid[0])
    summary = {}
    for spkid, name in CASES.items():
        d = np.load(ensure_mc(spkid))
        x = features_from_npz(d)[None, :]
        if cols is not None:
            x = x[:, cols]
        xn = torch.tensor((x - mu_x) / sd_x, dtype=torch.float32)
        dmin = np.asarray(d["dmin"])                     # (n_clones, 10)
        y = np.log10(np.clip(dmin, 1e-7, None))

        fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True)
        rows = []
        for w, ax in enumerate(axes.ravel()):
            yw = y[:, w]
            yw = yw[np.isfinite(yw)]
            F = ens_cdf(models, xn, torch.tensor([w]), grid)[0].numpy()
            Fm = np.maximum.accumulate(F) + np.linspace(0, 1e-9, len(F))
            qp = np.interp([0.05, 0.5, 0.95], Fm, grid.numpy())
            qt = np.quantile(yw, [0.05, 0.5, 0.95])
            Fe = (yw[None, :] <= grid.numpy()[:, None]).mean(1)
            crps = float(np.sum((F - Fe) ** 2) * dg)
            rows.append({"window": w, "crps": crps,
                         "q_pred": [float(v) for v in qp],
                         "q_mc": [float(v) for v in qt]})
            ax.hist(yw, bins=25, density=True, alpha=0.45, color="#4477aa",
                    label="MC (256)")
            ax.plot(grid.numpy(), np.gradient(F, dg), color="#e6774a", lw=1.8,
                    label="MDN ens.")
            ax.set_title(f"window {w} ({2025 + 10 * w}s)", fontsize=9)
            if w == 0:
                ax.legend(fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel(r"$\log_{10} d_{\min}$ [au]")
        fig.suptitle(f"{name} (SPK {spkid}): MC vs MDN ensemble, "
                     "10 decadal windows")
        fig.tight_layout()
        fig.savefig(FIGS / f"case_{name.lower()}.png")
        summary[name] = {"spkid": spkid, "windows": rows,
                         "mean_crps": float(np.mean([r["crps"] for r in rows]))}
        print(name, "mean CRPS", summary[name]["mean_crps"])
    (ROOT / "results" / "case_studies.json").write_text(
        json.dumps(summary, indent=2))
    print("saved results/case_studies.json and figures")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
