"""Fetch NEA orbital elements + covariance matrices from JPL SBDB (public API).

Stage 1: bulk query of all NEOs with orbital elements + uncertainties (sbdb_query.api).
Stage 2: per-object 6x6 covariance matrices (sbdb.api?cov=mat), with resume support.

Outputs:
  data/raw/nea_elements.csv        (bulk table)
  data/raw/cov/<spkid>.json        (per-object covariance, one file each -> resumable)
  logs/fetch_sbdb.log
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
COVDIR = RAW / "cov"
LOGDIR = ROOT / "logs"
for d in (RAW, COVDIR, LOGDIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGDIR / "fetch_sbdb.log"), logging.StreamHandler()],
)
log = logging.getLogger("fetch_sbdb")

QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"

FIELDS = [
    "spkid", "pdes", "full_name", "neo", "pha", "H",
    "epoch", "e", "a", "q", "i", "om", "w", "ma", "tp", "per_y", "moid",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om", "sigma_w", "sigma_ma",
    "condition_code", "n_obs_used", "data_arc",
]


def fetch_bulk() -> Path:
    out = RAW / "nea_elements.csv"
    if out.exists():
        log.info("bulk file exists, skip: %s", out)
        return out
    params = {
        "fields": ",".join(FIELDS),
        "sb-group": "neo",
        "full-prec": "true",
    }
    log.info("querying SBDB bulk API ...")
    j = None
    for attempt in range(10):
        try:
            r = requests.get(QUERY_URL, params=params, timeout=300)
            if r.status_code == 200:
                j = r.json()
                break
            log.warning("bulk query HTTP %d (attempt %d), retrying", r.status_code, attempt)
        except requests.RequestException as e:
            log.warning("bulk query attempt %d: %s", attempt, e)
        time.sleep(10)
    if j is None:
        raise RuntimeError("bulk query failed after retries")
    fields, data = j["fields"], j["data"]
    log.info("received %d NEOs", len(data))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(data)
    return out


def load_targets(bulk_csv: Path, max_cc: int, min_arc_days: float):
    """Select targets with reliable orbits (low condition code, long arc)."""
    import pandas as pd

    df = pd.read_csv(bulk_csv, low_memory=False)
    n0 = len(df)
    df = df.dropna(subset=["e", "a", "q", "i", "om", "w", "ma", "epoch", "moid"])
    df["condition_code"] = pd.to_numeric(df["condition_code"], errors="coerce")
    df["data_arc"] = pd.to_numeric(df["data_arc"], errors="coerce")
    df = df[(df["condition_code"] <= max_cc) & (df["data_arc"] >= min_arc_days)]
    # need uncertainties present for covariance to exist
    df = df.dropna(subset=["sigma_e", "sigma_a", "sigma_i"])
    log.info("target selection: %d -> %d (cc<=%d, arc>=%gd)", n0, len(df), max_cc, min_arc_days)
    return df


def fetch_cov(spkid: str, session: requests.Session, retries: int = 8) -> bool:
    out = COVDIR / f"{spkid}.json"
    if out.exists():
        return True
    for attempt in range(retries):
        try:
            r = session.get(SBDB_URL, params={"spk": spkid, "cov": "mat", "full-prec": "true"}, timeout=60)
            if r.status_code == 200:
                j = r.json()
                cov = j.get("orbit", {}).get("covariance")
                if cov is None:
                    out.write_text(json.dumps({"spkid": spkid, "covariance": None}))
                    return False
                keep = {
                    "spkid": spkid,
                    "covariance": cov,
                    "elements": j["orbit"].get("elements"),
                    "epoch": j["orbit"].get("epoch"),
                }
                out.write_text(json.dumps(keep))
                return True
            if r.status_code in (502, 503, 429):
                time.sleep(5 * (attempt + 1))
                continue
            log.warning("spk %s: HTTP %d", spkid, r.status_code)
            time.sleep(2)
        except requests.RequestException as e:
            log.warning("spk %s attempt %d: %s", spkid, attempt, e)
            time.sleep(5 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=3000)
    ap.add_argument("--max-cc", type=int, default=3)
    ap.add_argument("--min-arc-days", type=float, default=365.0)
    ap.add_argument("--moid-strata", action="store_true", default=True,
                    help="stratified sampling by MOID to over-represent close approachers")
    args = ap.parse_args()

    bulk = fetch_bulk()
    df = load_targets(bulk, args.max_cc, args.min_arc_days)

    # Stratified sampling by Earth MOID: all objects with moid<0.05 au (PHA-like),
    # fill the rest with a random sample of larger-MOID objects (seeded, reproducible).
    close = df[df["moid"] < 0.05]
    far = df[df["moid"] >= 0.05]
    n_far = max(0, args.n_targets - len(close))
    far_s = far.sample(n=min(n_far, len(far)), random_state=42)
    sel = close if len(close) >= args.n_targets else __import__("pandas").concat([close, far_s])
    sel = sel.head(max(args.n_targets, len(close)))
    log.info("selected %d targets (%d with MOID<0.05 au)", len(sel), len(close))
    sel.to_csv(RAW / "targets.csv", index=False)

    from concurrent.futures import ThreadPoolExecutor

    spkids = [str(int(s)) for s in sel["spkid"] if not (COVDIR / f"{int(s)}.json").exists()]
    log.info("%d covariances remaining to fetch", len(spkids))
    ok = fail = 0
    t0 = time.time()

    def work(spkid):
        s = requests.Session()
        return fetch_cov(spkid, s)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, res in enumerate(ex.map(work, spkids)):
            if res:
                ok += 1
            else:
                fail += 1
            if (k + 1) % 100 == 0:
                rate = (k + 1) / (time.time() - t0)
                eta = (len(spkids) - k - 1) / rate / 60
                log.info("progress %d/%d ok=%d fail=%d rate=%.2f/s eta=%.1f min",
                         k + 1, len(spkids), ok, fail, rate, eta)
    log.info("done: ok=%d fail=%d", ok, fail)


if __name__ == "__main__":
    main()
