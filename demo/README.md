# Transolver++ AirCraft — Full-Data Training Run (`demo/`)

A complete single-GPU training run of Transolver++ on **all available** AirCraft data
(30 aircraft geometries × 5 flow conditions = 150 surface-pressure samples), with an
independent train / val / test split. This folder is the committed showcase: it holds
the logs, the trained weights, the test predictions, the figures, and this report.

> Reproduce from the repo root with the conda `ml` environment. See "Commands" below.

---

## 1. Environment

| Item | Value |
|------|-------|
| OS | Windows 11 |
| Conda env | `ml` (`D:/Developer/miniconda3/envs/ml/python.exe`) |
| Python | 3.10.16 |
| PyTorch | 2.12.0+cu132 |
| GPU | NVIDIA RTX 5080, 16 GB, sm_120 (Blackwell) |

Single GPU: `WORLD_SIZE=1`, the process group is skipped and the eidetic-state
`all_reduce` is a no-op (see project `CLAUDE.md`).

## 2. Data & split

- **Source**: raw Tecplot FEPOINT surfaces `Components.i.dat` → `.h5` via
  `dataset/preprocess_dat_to_h5.py`, indexed by `result.csv`.
- **Per sample**: `pos (N,3)`, area-weighted `normals (N,3)`, `values (N,6)` =
  `[Cp, Rho, U, V, W, Pressure]`, attrs `Ma/alpha/beta`. **N = 331,971** nodes
  (identical surface topology across all samples).
- **Split by aircraft id** (held-out geometries, no same-aircraft leakage across splits):

  | split | aircraft ids | samples |
  |-------|--------------|---------|
  | train | the other 24 ids | 120 |
  | val   | 81, 82, 83 | 15 |
  | test  | 91, 92, 93 | 15 |

- **Normalization** (mean/std of pos / normals / values) is computed **from the train
  split only** and cached to `dataset/aircraft_dataset/norm_stats.json`, so val/test
  never leak into the statistics.

## 3. Model & optimization

```
Model(n_hidden=256, n_layers=4, space_dim=7, fun_dim=0, n_head=8,
      mlp_ratio=2, out_dim=6, slice_num=32, unified_pos=0, dropout=0.1)
# trainable parameters: 1,741,738
```

- **Inputs per node**: `x = [pos(3), sdf(1)=0, normals(3)]` → `space_dim=7`;
  global `condition = [Ma, alpha, beta]`.
- **Output**: 6 fields `[Cp, Rho, U, V, W, Pressure]` → `out_dim=6`.
  The reported metric is the **pressure** (last channel) relative L2 error (L2RE).
- **Optimizer**: Adam, `lr=1e-3`. **Scheduler**: OneCycleLR. **batch_size**: 1
  (one full mesh per step). **Epochs**: 50. `pos_norm=1`, `out_norm=1`.

## 4. Commands (reproduce)

```bash
# 1) preprocess all 30 ids -> 150 .h5 + train/val/test split json (120/15/15)
python dataset/preprocess_dat_to_h5.py --write_split --val_ids 81 82 83 --test_ids 91 92 93

# 2) train 50 epochs, validating on the val split every epoch; artifacts -> demo/
python main_airplane.py --nb_epochs 50 --val_iter 1 --out_path demo --dataset airplane \
    --cfd_model=transolver_plus \
    --data_dir dataset/aircraft_dataset/ --save_dir dataset/aircraft_dataset/

# 3) evaluate on the held-out test split (loads demo/model_50.pth)
python main_airplane.py --nb_epochs 50 --eval 1 --out_path demo --dataset airplane \
    --cfd_model=transolver_plus \
    --data_dir dataset/aircraft_dataset/ --save_dir dataset/aircraft_dataset/

# 4) figures (loss curve + per-epoch time + pred-vs-GT) and a phase-level profile
python scripts/plot_results.py --run_dir demo --data_dir dataset/aircraft_dataset
python -m scripts.profile_training --save_dir dataset/aircraft_dataset --n_train 10 --n_val 5
```

## 5. Results

| Metric | Value |
|--------|-------|
| Final train MSE loss (epoch 49) | 0.0215 |
| **Final validation pressure L2RE** (ids 81/82/83) | **0.1008** |
| **Mean test pressure L2RE** (ids 91/92/93, 15 samples) | **0.1058** |
| Test L2RE range | 0.088 – 0.133 |

Validation (0.1008) and test (0.1058) are on **disjoint, unseen aircraft** and agree
closely → the model generalizes to new geometries with no over-fit to the val set.

Per-sample test pressure L2RE:

| sample | L2RE | | sample | L2RE | | sample | L2RE |
|--------|------|--|--------|------|--|--------|------|
| 91_2_0_0 | 0.1329 | | 92_2_0_0 | 0.1029 | | 93_2_0_0 | 0.1163 |
| 91_7_0_0 | 0.1089 | | 92_7_0_0 | 0.0928 | | 93_7_0_0 | 0.1092 |
| 91_7_0_2 | 0.1097 | | 92_7_0_2 | 0.0945 | | 93_7_0_2 | 0.1099 |
| 91_7_7_0 | 0.1129 | | 92_7_7_0 | 0.0881 | | 93_7_7_0 | 0.1025 |
| 91_7_7_2 | 0.1122 | | 92_7_7_2 | 0.0895 | | 93_7_7_2 | 0.1044 |

### Figures (`demo/plots/`)
- **`loss_curve.png`** — train MSE loss (log), validation pressure L2RE, and per-epoch
  wall-clock time. Loss falls smoothly to ~0.02; val L2RE drops 0.65 → 0.10 with the
  usual early-LR jitter, then flattens.
- **`pred_vs_gt_91_7_0_0.png`** — surface pressure for the median-error test sample:
  Ground truth | Prediction | |error|. Prediction matches GT closely; residual error
  concentrates along high-gradient regions (leading edges / tip).

## 6. Timing breakdown & efficiency notes

Total wall-clock: **9012 s ≈ 2.5 h** for 50 epochs.

**Per-epoch wall time** (from `train.log` timestamps): median 109 s, but min **77 s** and
max **1613 s** (epoch 22) — a ~20× spread driven by a few spike epochs (15, 21, 22).

**Phase-level profile** of one train iteration (`demo/profile.txt`, RTX 5080, full mesh):

| phase | mean | share |
|-------|------|-------|
| data (h5 read + tensor build) | 9.2 ms | 1.5% |
| h2d (host→device) | 3.3 ms | 0.5% |
| fwd (forward) | 178 ms | 28.6% |
| **bwd (backward)** | **432 ms** | **69.2%** |
| opt (Adam + scheduler) | 1.6 ms | 0.3% |
| **total** | **0.62 s/iter** | 100% |

Steady-state epoch estimate `120×0.62 + 15×0.18 = 77.5 s` matches the observed fastest
epoch (76.6 s).

**Takeaways for improving training efficiency** (evidence-based, in priority order):

1. **Eliminate the wall-time spikes, not the data path.** Training is **compute-bound**:
   data loading is only 1.5%, so adding DataLoader workers / caching h5 in RAM would
   barely help. The ~85 min of "extra" time vs. the 65 min steady-state floor was lost
   to a handful of spike epochs (external GPU/system contention or thermal throttling).
   Running with exclusive GPU access removes the single biggest inefficiency.
2. **Mixed precision (AMP/bf16).** Backward dominates (69%); the RTX 5080's bf16 tensor
   cores can plausibly cut fwd+bwd by ~1.5–2× via `torch.autocast` + `GradScaler`.
   Memory headroom exists (peak ~9.9 / 16 GB), so gradient checkpointing is unnecessary.
3. **`torch.compile`** to fuse the per-block attention/MLP kernels (helps both fwd/bwd).
4. **Larger effective batch** is hard at full mesh (one sample ≈ 9.9 GB peak); point
   subsampling (`--r` / `split_size`) would be needed to batch multiple geometries.

## 7. Files in this folder

| file | description |
|------|-------------|
| `train.log` | per-epoch train_loss / val_loss_mse / val_loss_l2re (timestamped) |
| `test.log` | per-sample + mean test pressure L2RE, with reference coefficients |
| `log_50.json` | final summary (params, time, hparams, final losses) |
| `model_50.pth`, `model_49.pth` | trained model objects (`torch.load(..., weights_only=False)`) |
| `output/*.npy` | per-test-sample predictions, shape `(1, N, 6)`, denormalized |
| `plots/loss_curve.png`, `plots/pred_vs_gt_*.png` | figures |
| `profile.txt` | raw phase-level profiling output |
