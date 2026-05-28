"""Visualize a Transolver++ AirCraft training run.

Reads the artifacts produced by main_airplane.py (train.log, test.log, the per-sample
prediction .npy files) and produces:

  1. <run_dir>/plots/loss_curve.png      - train MSE loss and val L2RE vs epoch
  2. <run_dir>/plots/pred_vs_gt_<name>.png - 3D surface pressure: GT | Pred | |error|

It also prints the final validation L2RE (last epoch in train.log) and the mean test
L2RE (from test.log), and selects the median-error test sample for the comparison plot.

Usage:
    python scripts/plot_results.py --run_dir demo --data_dir dataset/aircraft_dataset
"""
import argparse
import datetime
import os
import re

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+),\s*train_loss:\s*([\d.eE+-]+)"
    r"(?:.*?val_loss_mse:\s*([\d.eE+-]+),\s*val_loss_l2re:\s*([\d.eE+-]+))?"
)
SAMPLE_RE = re.compile(r"(\S+):\s*pressure L2RE=([\d.eE+-]+)")
TS_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)\s*-\s*Epoch\s+(\d+),")


def parse_epoch_times(log_path):
    """Per-epoch wall-clock seconds from train.log timestamps (epoch i -> t_i - t_{i-1})."""
    epochs, stamps = [], []
    with open(log_path, "r") as f:
        for line in f:
            m = TS_RE.search(line)
            if not m:
                continue
            t = (datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                 + datetime.timedelta(milliseconds=int(m.group(2))))
            epochs.append(int(m.group(3)))
            stamps.append(t)
    secs = [(stamps[i] - stamps[i - 1]).total_seconds() for i in range(1, len(stamps))]
    return np.array(epochs[1:]), np.array(secs)


def parse_train_log(log_path):
    """Return per-epoch arrays (epoch, train_loss, val_mse, val_l2re).

    val_* are NaN on epochs where validation did not run.
    """
    epochs, train_loss, val_mse, val_l2re = [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_mse.append(float(m.group(3)) if m.group(3) else np.nan)
            val_l2re.append(float(m.group(4)) if m.group(4) else np.nan)
    return (np.array(epochs), np.array(train_loss),
            np.array(val_mse), np.array(val_l2re))


def parse_test_log(log_path):
    """Return {sample_name: l2re} and the mean L2RE printed by the eval run."""
    per_sample = {}
    mean_l2re = None
    with open(log_path, "r") as f:
        for line in f:
            m = SAMPLE_RE.search(line)
            if m:
                per_sample[m.group(1)] = float(m.group(2))
                continue
            mm = re.search(r"Average pressure L2RE over \d+ test samples:\s*([\d.eE+-]+)", line)
            if mm:
                mean_l2re = float(mm.group(1))
    return per_sample, mean_l2re


def plot_loss_curve(run_dir, out_png):
    epochs, train_loss, _, val_l2re = parse_train_log(os.path.join(run_dir, "train.log"))
    if len(epochs) == 0:
        raise RuntimeError(f"no 'Epoch ...' lines parsed from {run_dir}/train.log")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.5))
    ax1.plot(epochs, train_loss, color="tab:blue", marker=".", ms=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train MSE loss (log)")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)

    mask = ~np.isnan(val_l2re)
    ax2.plot(epochs[mask], val_l2re[mask], color="tab:red", marker=".", ms=3)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation pressure L2RE")
    ax2.set_title("Validation error")
    ax2.grid(True, alpha=0.3)

    te, tsec = parse_epoch_times(os.path.join(run_dir, "train.log"))
    if len(tsec):
        med = float(np.median(tsec))
        ax3.bar(te, tsec, color="tab:green", alpha=0.7)
        ax3.axhline(med, color="k", ls="--", lw=1, label=f"median {med:.0f}s")
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("wall-clock seconds")
        ax3.set_title("Per-epoch time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


def plot_pred_vs_gt(h5_path, npy_path, out_png, max_points=40000, seed=0):
    with h5py.File(h5_path, "r") as f:
        pos = f["pos"][:]
        gt = f["values"][:, -1]  # pressure is the last channel
    pred = np.load(npy_path)[0, :, -1]  # (1, N, 6) -> (N,) pressure

    n = pos.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_points, replace=False)
        pos, gt, pred = pos[idx], gt[idx], pred[idx]

    err = np.abs(pred - gt)
    vmin, vmax = float(min(gt.min(), pred.min())), float(max(gt.max(), pred.max()))

    fig = plt.figure(figsize=(16, 5))
    specs = [("Ground truth", gt, "viridis", vmin, vmax),
             ("Prediction", pred, "viridis", vmin, vmax),
             ("|error|", err, "inferno", 0.0, float(err.max()))]
    for i, (title, c, cmap, lo, hi) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, cmap=cmap,
                        vmin=lo, vmax=hi, s=2, marker=".")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=28, azim=-62)
        # honest box aspect, but clamp the slender (thickness) axis so the surface
        # is not rendered edge-on as a thin sliver.
        try:
            asp = np.array([np.ptp(pos[:, j]) for j in range(3)], dtype=float)
            asp = np.clip(asp / asp.max(), 0.25, 1.0)
            ax.set_box_aspect(tuple(asp))
        except Exception:
            pass
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)

    name = os.path.splitext(os.path.basename(h5_path))[0]
    fig.suptitle(f"Surface pressure — sample {name}", y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="demo")
    ap.add_argument("--data_dir", default="dataset/aircraft_dataset")
    ap.add_argument("--sample", default=None,
                    help="test sample name to plot (default: median-error sample)")
    args = ap.parse_args()

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) loss curve + final validation error
    epochs, _, _, val_l2re = parse_train_log(os.path.join(args.run_dir, "train.log"))
    final_val = next((v for v in reversed(val_l2re) if not np.isnan(v)), np.nan)
    plot_loss_curve(args.run_dir, os.path.join(plots_dir, "loss_curve.png"))

    # 2) test errors + pick the comparison sample
    per_sample, mean_l2re = parse_test_log(os.path.join(args.run_dir, "test.log"))
    print(f"\nFinal validation pressure L2RE: {final_val:.4f}")
    if mean_l2re is not None:
        print(f"Mean test pressure L2RE ({len(per_sample)} samples): {mean_l2re:.4f}")

    if args.sample:
        chosen = args.sample
    elif per_sample:
        ordered = sorted(per_sample.items(), key=lambda kv: kv[1])
        chosen = ordered[len(ordered) // 2][0]  # median-error sample
    else:
        chosen = None

    if chosen:
        h5_path = os.path.join(args.data_dir, f"{chosen}.h5")
        npy_path = os.path.join(args.run_dir, "output", f"{chosen}.npy")
        if os.path.exists(h5_path) and os.path.exists(npy_path):
            print(f"comparison sample: {chosen} (L2RE={per_sample.get(chosen, float('nan')):.4f})")
            plot_pred_vs_gt(h5_path, npy_path,
                            os.path.join(plots_dir, f"pred_vs_gt_{chosen}.png"))
        else:
            print(f"WARNING: missing {h5_path} or {npy_path}; skipping comparison plot")


if __name__ == "__main__":
    main()
