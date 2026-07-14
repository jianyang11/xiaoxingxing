"""Train the MDN surrogate (deep ensemble via --seed).

Resumable: checkpoints/mdn_seed<k>.pt stores model+optimizer+epoch; training
continues from the last checkpoint if present.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from model import MDN, N_WINDOWS

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
LOGDIR = ROOT / "logs"
CKPT.mkdir(exist_ok=True)


def setup_log(name):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOGDIR / f"{name}.log"),
                  logging.StreamHandler()])
    return logging.getLogger(name)


def load_data(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for split in ("train", "val", "test"):
        if f"{split}_x" not in d:
            continue
        out[split] = (d[f"{split}_x"], d[f"{split}_y"], d[f"{split}_spk"])
    return out


def make_pairs(x, y):
    """Expand (asteroid, clone, window) -> flat sample arrays."""
    n_ast, n_clone, n_win = y.shape
    xs, ws, ys = [], [], []
    for w in range(n_win):
        yy = y[:, :, w]                      # (n_ast, n_clone)
        mask = ~np.isnan(yy)
        idx_ast, idx_clone = np.where(mask)
        xs.append(x[idx_ast])
        ws.append(np.full(mask.sum(), w))
        ys.append(yy[mask])
    return (np.concatenate(xs), np.concatenate(ws), np.concatenate(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--comp", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--diag-cov", action="store_true",
                    help="use only the 6 diagonal Cholesky terms of covariance")
    args = ap.parse_args()

    log = setup_log(f"train_seed{args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    data = load_data(args.data)
    xtr, ytr, _ = data["train"]
    xva, yva, _ = data["val"]
    if args.diag_cov:
        n_el = xtr.shape[1] - 21
        cols = list(range(n_el)) + [n_el + k for k in (0, 2, 5, 9, 14, 20)]
        xtr, xva = xtr[:, cols], xva[:, cols]

    # normalize features with train stats
    mu_x, sd_x = xtr.mean(0), xtr.std(0) + 1e-8
    Xtr, Wtr, Ytr = make_pairs((xtr - mu_x) / sd_x, ytr)
    Xva, Wva, Yva = make_pairs((xva - mu_x) / sd_x, yva)
    log.info("train samples %d, val samples %d", len(Ytr), len(Yva))

    dev = "cpu"
    model = MDN(Xtr.shape[1], args.hidden, args.blocks, args.comp,
                args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ck = CKPT / f"mdn_seed{args.seed}.pt"
    start_ep, best_val = 0, np.inf
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep, best_val = st["epoch"] + 1, st["best_val"]
        log.info("resumed from epoch %d (best val %.4f)", start_ep, best_val)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Wtr_t = torch.tensor(Wtr, dtype=torch.long)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Wva_t = torch.tensor(Wva, dtype=torch.long)
    Yva_t = torch.tensor(Yva, dtype=torch.float32)

    n = len(Ytr_t)
    for ep in range(start_ep, args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, args.bs):
            j = perm[i:i + args.bs]
            loss = model.nll(Xtr_t[j], Wtr_t[j], Ytr_t[j])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(Yva_t), 65536):
                vl += model.nll(Xva_t[i:i+65536], Wva_t[i:i+65536],
                                Yva_t[i:i+65536]).item() * min(65536, len(Yva_t)-i)
            vl /= len(Yva_t)
        log.info("ep %d train %.4f val %.4f (%.1fs)", ep, tot / n, vl,
                 time.time() - t0)
        best_val = min(best_val, vl)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_val": best_val, "mu_x": mu_x, "sd_x": sd_x,
                    "args": vars(args)}, ck)
        if vl <= best_val:
            torch.save({"model": model.state_dict(), "mu_x": mu_x, "sd_x": sd_x,
                        "args": vars(args), "epoch": ep, "val": vl},
                       CKPT / f"mdn_seed{args.seed}_best.pt")


if __name__ == "__main__":
    main()
