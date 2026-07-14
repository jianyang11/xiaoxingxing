"""Clone Monte-Carlo propagation of NEAs with REBOUND (ground-truth generator).

Physics:
  - Sun + 8 planets + Moon, initial states from NASA Horizons at JD 2461000.5
    (cached in data/raw/planets_2461000.5.bin, ecliptic J2000 frame).
  - Asteroid clones are massless test particles sampled from the SBDB 6x6
    covariance (cometary elements e, q, tp, node, peri, i at the covariance epoch).
  - Integrator IAS15 (adaptive, machine precision, robust through close encounters).
  - Units: G=1, lengths in au, time in yr/(2*pi).

Output per asteroid (npz): for each clone, minimum Earth distance within each
10-yr window over 100 yr, plus global minimum and its epoch; refined by
quadratic interpolation around sampled minima.
"""
import json
import numpy as np
import rebound
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANETS_BIN = ROOT / "data" / "raw" / "planets_2461000.5.bin"
EPOCH0_JD = 2461000.5           # sim t=0
TWOPI = 2.0 * np.pi
YR = TWOPI                      # 1 yr in sim units
DAY = TWOPI / 365.25

COV_LABELS = ["e", "q", "tp", "node", "peri", "i"]  # SBDB cometary order


def jd_to_t(jd: float) -> float:
    return (jd - EPOCH0_JD) / 365.25 * YR


def load_cov(path: Path):
    d = json.loads(Path(path).read_text())
    c = d["covariance"]
    if c is None:
        return None
    labels = c["labels"]
    idx = [labels.index(l) for l in COV_LABELS]
    M = np.array(c["data"], dtype=float)[np.ix_(idx, idx)]
    els = {e["name"]: float(e["value"]) for e in c["elements"]}
    mean = np.array([els["e"], els["q"], els["tp"], els["om"], els["w"], els["i"]])
    return {"mean": mean, "cov": M, "epoch_jd": float(c["epoch"]), "spkid": d["spkid"]}


def sample_clones(mean, cov, n, rng):
    """Sample clones; covariance can be near-singular -> use eigval clipping."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    A = V * np.sqrt(w)
    z = rng.standard_normal((n, 6))
    samples = mean + z @ A.T
    # physical sanity: e in (0,1) for bound elliptic, q>0
    samples[:, 0] = np.clip(samples[:, 0], 1e-8, 0.9999)
    samples[:, 1] = np.clip(samples[:, 1], 1e-6, None)
    return samples


def build_sim(epoch_jd: float) -> rebound.Simulation:
    sim = rebound.Simulation(str(PLANETS_BIN))
    sim.integrator = "ias15"
    sim.move_to_com()
    sim.integrate(jd_to_t(epoch_jd))
    return sim


def add_clones(sim, clones, epoch_jd):
    sun = sim.particles[0]
    t = jd_to_t(epoch_jd)
    for e, q, tp, node, peri, inc in clones:
        a = q / (1.0 - e)
        sim.add(primary=sun, a=a, e=e, inc=np.radians(inc),
                Omega=np.radians(node), omega=np.radians(peri),
                T=jd_to_t(tp), m=0.0)
    sim.N_active = 10  # planets+Sun+Moon are massive; clones are test particles
    return sim


def propagate(sim, n_clones, t_end_yr=100.0, sample_days=1.8, earth_index=3,
              n_windows=10):
    """Integrate and track per-clone min Earth distance in decade windows."""
    t0 = sim.t
    t_end = t0 + t_end_yr * YR
    dt_out = sample_days * DAY
    nsteps = int(np.ceil((t_end - t0) / dt_out))
    win_len = t_end_yr * YR / n_windows

    npl = 11  # Sun..Neptune (10 massive) -- indices 0..9, clones from 10
    n_massive = 10
    dmin = np.full((n_clones, n_windows), np.inf)
    tmin = np.zeros((n_clones, n_windows))
    prev = np.full((n_clones, 3), np.nan)  # last three distance samples for refine
    # store last two distances for quadratic refinement
    d_hist = np.full((3, n_clones), np.inf)
    t_hist = np.zeros(3)

    xyz = np.zeros((n_massive + n_clones, 3))
    for k in range(1, nsteps + 1):
        t_target = min(t0 + k * dt_out, t_end)
        sim.integrate(t_target)
        a = sim.serialize_particle_data(xyz=xyz)
        earth = xyz[earth_index]
        d = np.linalg.norm(xyz[n_massive:] - earth, axis=1)
        d_hist = np.roll(d_hist, -1, axis=0); d_hist[-1] = d
        t_hist = np.roll(t_hist, -1); t_hist[-1] = sim.t
        w = min(int((sim.t - t0) / win_len), n_windows - 1)
        # local minimum refinement (parabola through last 3 samples)
        if k >= 3:
            mid = d_hist[1]
            is_min = (mid < d_hist[0]) & (mid < d_hist[2])
            if is_min.any():
                y0, y1, y2 = d_hist[0], d_hist[1], d_hist[2]
                denom = (y0 - 2 * y1 + y2)
                denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
                delta = 0.5 * (y0 - y2) / denom  # in units of dt_out
                d_ref = y1 - 0.125 * (y0 - y2) ** 2 / denom
                d_ref = np.where(is_min, np.minimum(d_ref, mid), np.inf)
                wm = min(int((t_hist[1] - t0) / win_len), n_windows - 1)
                upd = d_ref < dmin[:, wm]
                dmin[upd, wm] = d_ref[upd]
                tmin[upd, wm] = t_hist[1] + delta[upd] * dt_out
        # also plain sample update (covers window edges / monotonic segments)
        upd = d < dmin[:, w]
        dmin[upd, w] = d[upd]
        tmin[upd, w] = sim.t
    return dmin, tmin


def run_one(cov_json_path: Path, n_clones: int, seed: int, out_path: Path,
            t_end_yr=100.0, sample_days=1.8):
    info = load_cov(cov_json_path)
    if info is None:
        return False
    rng = np.random.default_rng(seed)
    clones = sample_clones(info["mean"], info["cov"], n_clones, rng)
    sim = build_sim(info["epoch_jd"])
    add_clones(sim, clones, info["epoch_jd"])
    dmin, tmin = propagate(sim, n_clones, t_end_yr=t_end_yr, sample_days=sample_days)
    np.savez_compressed(
        out_path,
        spkid=info["spkid"], epoch_jd=info["epoch_jd"],
        mean=info["mean"], cov=info["cov"], clones=clones,
        dmin=dmin, tmin_sim=tmin, t_end_yr=t_end_yr, sample_days=sample_days,
        seed=seed,
    )
    return True


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("cov_json")
    ap.add_argument("--n-clones", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/test_prop.npz")
    ap.add_argument("--years", type=float, default=100.0)
    ap.add_argument("--sample-days", type=float, default=1.8)
    a = ap.parse_args()
    t0 = time.time()
    ok = run_one(Path(a.cov_json), a.n_clones, a.seed, Path(a.out),
                 t_end_yr=a.years, sample_days=a.sample_days)
    print("ok=%s wall=%.1fs" % (ok, time.time() - t0))
