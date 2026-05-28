"""Phase-level timing breakdown for a Transolver++ AirCraft training step.

Reuses the exact dataset / model / normalization path of main_airplane.py and times
each phase of a few train and validation iterations to locate the bottleneck:

  data   : DataLoader yield (h5 read from disk + tensor construct + torch.cat)
  h2d    : host->device transfers (.to(device))
  fwd    : model forward
  bwd    : loss + backward
  opt    : optimizer.step + scheduler.step
  val    : validation forward (no grad)

GPU phases are wrapped in torch.cuda.synchronize() so the wall-clock reflects real
compute, not async-launch overhead.

Usage:
    python scripts/profile_training.py --save_dir dataset/aircraft_dataset --n_train 8 --n_val 4
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler

from dataset.dataset import AirplaneDataLoader, AirplaneDataset, load_or_compute_norm_stats
from models.Transolver_plus import Model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="dataset/aircraft_dataset")
    ap.add_argument("--n_train", type=int, default=8, help="train iters to time (after 1 warmup)")
    ap.add_argument("--n_val", type=int, default=4, help="val iters to time")
    args = ap.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)

    train_dataset = AirplaneDataset(args.save_dir, split="train")
    val_dataset = AirplaneDataset(args.save_dir, split="val")
    stats = load_or_compute_norm_stats(args.save_dir, train_dataset.f_list)
    pos_mean, pos_std = stats["pos_mean"].cuda(), stats["pos_std"].cuda()
    out_mean, out_std = stats["out_mean"].cuda(), stats["out_std"].cuda()

    train_loader = AirplaneDataLoader(
        train_dataset, batch_size=1,
        sampler=RandomSampler(train_dataset, generator=torch.Generator().manual_seed(0)))
    val_loader = AirplaneDataLoader(
        val_dataset, batch_size=1,
        sampler=RandomSampler(val_dataset, generator=torch.Generator().manual_seed(0)))

    model = Model(n_hidden=256, n_layers=4, space_dim=7, fun_dim=0, n_head=8,
                  mlp_ratio=2, out_dim=6, slice_num=32, unified_pos=0, dropout=0.1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10000)
    criterion = nn.MSELoss(reduction="none")

    acc = {k: [] for k in ("data", "h2d", "fwd", "bwd", "opt")}
    h5_mb = []

    # ---- train phase ----
    model.train()
    it = iter(train_loader)
    for i in range(args.n_train + 1):  # +1 warmup (first GPU kernels compile)
        t0 = time.perf_counter()
        x, y, pos, geom, _ = next(it)
        sync()
        t1 = time.perf_counter()

        x = x.to(device)
        pos = pos.to(device)
        y = y.to(device)
        geom = geom.to(device)
        sync()
        t2 = time.perf_counter()

        pos_n = (pos - pos_mean) / pos_std
        x[:, :, :3] = pos_n
        optimizer.zero_grad()
        out = model((x, pos_n, geom))
        sync()
        t3 = time.perf_counter()

        y_n = (y - out_mean) / (out_std + 1e-6)
        loss = criterion(out, y_n).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sync()
        t4 = time.perf_counter()

        optimizer.step()
        scheduler.step()
        sync()
        t5 = time.perf_counter()

        if i == 0:
            continue  # discard warmup
        acc["data"].append(t1 - t0)
        acc["h2d"].append(t2 - t1)
        acc["fwd"].append(t3 - t2)
        acc["bwd"].append(t4 - t3)
        acc["opt"].append(t5 - t4)
        h5_mb.append(x.numel() * 4 / 1e6 + y.numel() * 4 / 1e6)

    # ---- val phase ----
    val_times = []
    model.eval()
    with torch.no_grad():
        vit = iter(val_loader)
        for j in range(args.n_val):
            t0 = time.perf_counter()
            x, y, pos, geom, _ = next(vit)
            x = x.to(device)
            pos = pos.to(device)
            y = y.to(device)
            geom = geom.to(device)
            pos_n = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos_n
            _ = model((x, pos_n, geom))
            sync()
            val_times.append(time.perf_counter() - t0)

    # ---- report ----
    def m(v):
        return (np.mean(v), np.std(v))

    print(f"\nDevice: {torch.cuda.get_device_name(0)}  |  points/sample: {pos.shape[1]}")
    print(f"Timed {args.n_train} train iters (1 warmup discarded), {args.n_val} val iters\n")
    per_iter = sum(np.mean(acc[k]) for k in acc)
    print(f"{'phase':<8}{'mean (s)':>12}{'std':>10}{'% of train-iter':>18}")
    for k in ("data", "h2d", "fwd", "bwd", "opt"):
        mean, std = m(acc[k])
        print(f"{k:<8}{mean:>12.4f}{std:>10.4f}{100 * mean / per_iter:>17.1f}%")
    print(f"{'TOTAL':<8}{per_iter:>12.4f}{'':>10}{'100.0%':>18}")
    vmean, vstd = m(val_times)
    print(f"\nval-iter mean: {vmean:.4f}s (std {vstd:.4f})")
    print(f"per-sample tensor payload (x+y): {np.mean(h5_mb):.1f} MB")

    n_train, n_val = len(train_dataset), len(val_dataset)
    est = per_iter * n_train + vmean * n_val
    print(f"\nEstimated steady-state epoch: {n_train} train x {per_iter:.2f}s "
          f"+ {n_val} val x {vmean:.2f}s = {est:.1f}s ({est / 60:.1f} min)")
    data_share = 100 * np.mean(acc["data"]) / per_iter
    print(f"Data-loading share of a train iter: {data_share:.1f}%  "
          f"({'I/O-bound -> add DataLoader workers / cache h5 in RAM' if data_share > 30 else 'compute-bound'})")


if __name__ == "__main__":
    main()
