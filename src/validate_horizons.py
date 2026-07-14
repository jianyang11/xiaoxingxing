"""Cross-validate our REBOUND propagation against JPL Horizons ephemerides.

For a set of well-known NEAs: take heliocentric state from Horizons at epoch T0,
propagate with our dynamical model (Sun+8 planets+Moon, IAS15) for DT years,
compare with Horizons position at T0+DT. Report position error in km and in
units of Earth radii / lunar distance.
"""
import numpy as np
import rebound
import reboundx
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANETS_BIN = ROOT / "data" / "raw" / "planets_2461000.5.bin"
EPOCH0_JD = 2461000.5
TWOPI = 2 * np.pi
AU_KM = 149597870.7


def jd_to_t(jd):
    return (jd - EPOCH0_JD) / 365.25 * TWOPI


def state_from_horizons(name, jd):
    s = rebound.Simulation()
    s.add(name, date="JD%.1f" % jd)
    p = s.particles[0]
    return np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz])


def run(name, jd0, dt_years, gr=True):
    # Horizons gives heliocentric(?) - rebound adds relative to sim COM if empty sim:
    # with a single particle it's the barycentric state in ecliptic frame.
    sim = rebound.Simulation(str(PLANETS_BIN))
    sim.integrator = "ias15"
    if gr:
        rx = reboundx.Extras(sim)
        f = rx.load_force("gr")
        f.params["c"] = 10065.32  # speed of light in au/(yr/2pi)
        rx.add_force(f)
    # planets bin is raw Horizons barycentric-ish states (added sequentially);
    # keep frame consistent: do NOT move_to_com for this comparison; Horizons
    # states for planets and asteroid are all SSB-frame vectors.
    sim.integrate(jd_to_t(jd0))
    st = state_from_horizons(name, jd0)
    sim.add(x=st[0], y=st[1], z=st[2], vx=st[3], vy=st[4], vz=st[5], m=0.0)
    sim.N_active = 10
    jd1 = jd0 + dt_years * 365.25
    sim.integrate(jd_to_t(jd1))
    p = sim.particles[-1]
    ours = np.array([p.x, p.y, p.z])
    ref = state_from_horizons(name, jd1)[:3]
    err_km = np.linalg.norm(ours - ref) * AU_KM
    return err_km


if __name__ == "__main__":
    targets = ["Apophis", "Bennu", "433"]  # Eros
    jd0 = 2461000.5
    for gr in (True, False):
        print("=== GR:", gr)
        for dt in (1.0, 5.0, 10.0):
            for name in targets:
                try:
                    e = run(name, jd0, dt, gr=gr)
                    print(f"{name:10s} dt={dt:5.1f}yr  err={e:12.1f} km  ({e/6371:.1f} R_E, {e/384400:.4f} LD)")
                except Exception as ex:
                    print(name, dt, "FAILED", ex)
                time.sleep(2)
