"""Build the learning dataset from simulated clone-MC outputs.

Input per asteroid (data/sim/<split>/<spkid>.npz):
  mean (6,)  cometary elements [e, q, tp_jd, node_deg, peri_deg, i_deg] at cov epoch
  cov (6,6)  covariance
  dmin (n_clones, 10)  min Earth distance per 10-yr window (au)

Feature vector x (per asteroid, 27 dims):
  e, q, log(a), sin/cos of node, peri, i, (tp - epoch) mod P / P phase sin/cos,
  scaled Cholesky of covariance (21 log-scaled entries)

Target per (asteroid, window): the set of clone log10(dmin) values.
The MDN is trained on individual clone samples: (x, window) -> log10 dmin.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIMDIR = ROOT / "data" / "sim"

N_WINDOWS = 10
EPS = 1e-12


def features_from_npz(d):
    e, q, tp, node, peri, inc = d["mean"]
    epoch = float(d["epoch_jd"])
    a = q / (1.0 - e)
    P_days = 365.25 * a ** 1.5
    phase = 2 * np.pi * (((tp - epoch) / P_days) % 1.0)
    ang = np.radians([node, peri, inc])
    f_el = [e, q, np.log(a),
            np.sin(ang[0]), np.cos(ang[0]),
            np.sin(ang[1]), np.cos(ang[1]),
            np.sin(ang[2]), np.cos(ang[2]),
            np.sin(phase), np.cos(phase)]
    C = np.array(d["cov"], dtype=float)
    # scale rows/cols to natural units: tp in days ~ ok; angles deg ~ ok
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0, None)
    L = np.linalg.cholesky(C + np.eye(6) * (1e-30 + w.max() * 1e-12))
    tri = L[np.tril_indices(6)]
    f_cov = np.sign(tri) * np.log10(np.abs(tri) + EPS)  # signed log scale
    return np.array(f_el + list(f_cov), dtype=np.float64)


def load_split(split):
    xs, ys, spk = [], [], []
    for f in sorted((SIMDIR / split).glob("*.npz")):
        d = np.load(f)
        x = features_from_npz(d)
        dmin = np.asarray(d["dmin"])           # (n_clones, 10)
        y = np.log10(np.clip(dmin, 1e-7, None))
        xs.append(x)
        ys.append(y)
        spk.append(str(d["spkid"]))
    return xs, ys, spk


def build(out=ROOT / "data" / "dataset.npz"):
    data = {}
    for split in ("train", "val", "test"):
        xs, ys, spk = load_split(split)
        if not xs:
            continue
        data[f"{split}_x"] = np.stack(xs)
        # ys have differing n_clones between splits; store as object? pad instead
        nmax = max(y.shape[0] for y in ys)
        Y = np.full((len(ys), nmax, N_WINDOWS), np.nan)
        for i, y in enumerate(ys):
            Y[i, :y.shape[0]] = y
        data[f"{split}_y"] = Y
        data[f"{split}_spk"] = np.array(spk)
    np.savez_compressed(out, **data)
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
    return out


if __name__ == "__main__":
    build()
