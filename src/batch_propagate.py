"""Batch clone-MC propagation over all fetched covariances.

- Resumable: output npz per asteroid in data/sim/<split>/<spkid>.npz; skips existing.
- Parallel: multiprocessing Pool (2 workers on this box).
- Split assignment: deterministic hash of spkid -> train/val/test (80/8/12).
- Test split gets more clones (256) for tighter ground-truth distributions.

Usage: python src/batch_propagate.py [--workers 2]
"""
import argparse
import hashlib
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COVDIR = ROOT / "data" / "raw" / "cov"
SIMDIR = ROOT / "data" / "sim"
LOGDIR = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGDIR / "batch_propagate.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("batch")

N_CLONES = {"train": 128, "val": 128, "test": 256}


def split_of(spkid: str) -> str:
    h = int(hashlib.sha256(spkid.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 88:
        return "val"
    return "test"


def job_list():
    jobs = []
    for f in sorted(COVDIR.glob("*.json")):
        spkid = f.stem
        try:
            c = json.loads(f.read_text()).get("covariance")
            if c is None or "elements" not in c:
                continue
        except Exception:
            continue
        sp = split_of(spkid)
        out = SIMDIR / sp / f"{spkid}.npz"
        if not out.exists():
            jobs.append((str(f), sp, str(out)))
    return jobs


def run_job(args):
    cov_path, sp, out = args
    from propagate import run_one  # import in worker
    spkid = Path(cov_path).stem
    seed = int(spkid) % (2**31)
    t0 = time.time()
    try:
        tmp = out + ".tmp.npz"
        ok = run_one(Path(cov_path), N_CLONES[sp], seed, Path(tmp))
        if ok:
            os.replace(tmp, out)
        return (spkid, ok, time.time() - t0, None)
    except Exception as e:
        return (spkid, False, time.time() - t0, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--loop", action="store_true",
                    help="keep polling for new covariance files (while fetch runs)")
    args = ap.parse_args()
    for sp in N_CLONES:
        (SIMDIR / sp).mkdir(parents=True, exist_ok=True)

    while True:
        jobs = job_list()
        if not jobs:
            if args.loop and any(p.name == "fetch_sbdb.py" for p in []):
                time.sleep(60)
                continue
            # check whether fetch is still running
            import subprocess
            r = subprocess.run(["pgrep", "-f", "fetch_sbdb.py"], capture_output=True)
            if args.loop and r.returncode == 0:
                log.info("no jobs; fetch still running, sleeping 120s")
                time.sleep(120)
                continue
            log.info("all done")
            break
        log.info("starting %d jobs with %d workers", len(jobs), args.workers)
        done = 0
        t0 = time.time()
        with Pool(args.workers, maxtasksperchild=1) as pool:
            for spkid, ok, wall, err in pool.imap_unordered(run_job, jobs):
                done += 1
                if err:
                    log.warning("spk %s FAILED: %s", spkid, err)
                if done % 10 == 0:
                    rate = done / (time.time() - t0)
                    log.info("progress %d/%d rate=%.2f/min eta=%.1f h",
                             done, len(jobs), rate * 60,
                             (len(jobs) - done) / rate / 3600)
        if not args.loop:
            break


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
