# Transolver++ — Project Review & Usage Guide

This document summarizes the project configuration, the standard AirCraft workflow, the
debugging/fixes applied to make it runnable on a local single-GPU Windows machine, the
verified test results, and how to migrate the pipeline to a new dataset.

> Paper: *Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries*
> (ICML 2025). Source PDF: `references/2502.02414v2.pdf`.

---

## 1. What this project is

Transolver++ is a neural PDE solver that scales attention over **million-scale** surface
meshes. The core idea (vs. the original Transolver):

- **Eidetic states** — Gumbel-Softmax slice reparameterization (`gumbel_softmax`, Rep-Slice)
  plus a learned local-adaptive temperature (`proj_temperature`, Ada-Temp). Prevents the
  physical-state degeneration that plain softmax slicing suffers at large mesh sizes.
- **Parallel framework** — eidetic states are aggregated across GPUs with an `all_reduce`
  whose communication cost is independent of mesh size (paper Algorithm 1).
- **Memory saving** — drops the redundant `f` projection branch of Transolver.

This repository contains the **AirCraft industrial track only** (3D aircraft surface-pressure
regression). The standard-benchmark experiments (Elasticity/Airfoil/Pipe/etc.) are *not*
included as runnable code — see §7.

---

## 2. Environment & configuration

| Item | Value (verified) |
|------|------------------|
| OS | Windows 11 |
| Conda env | `ml` |
| Python | 3.10.16 |
| PyTorch | 2.12.0+cu132 |
| GPU | NVIDIA RTX 5080, 16 GB, sm_120 |
| Key deps | einops, timm, h5py, tqdm, numpy, pandas, pyyaml, scipy |

Model configuration (single GPU), set in `main_airplane.py`:

```
Model(n_hidden=256, n_layers=4, space_dim=7, fun_dim=0, n_head=8,
      mlp_ratio=2, out_dim=6, slice_num=32, unified_pos=0, dropout=0.1)
# ~1.74M parameters
```

Inputs per node: `x = [pos(3), sdf(1)=0, normals(3)]` → `space_dim=7`; global `condition = [Ma, alpha, beta]`.
Outputs: 6 physical fields `[Cp, Rho, U, V, W, Pressure]` → `out_dim=6`.

---

## 3. Key files

| File | Role |
|------|------|
| `models/Transolver_plus.py` | Model; `Physics_Attention_1D_Eidetic` is the core block |
| `main_airplane.py` | Entry point: args, optional dist init, normalization, train/eval dispatch |
| `train_airplane.py` | Training loop + validation (`train.main`, `train.test`) |
| `dataset/dataset.py` | `AirplaneDataset`/`AirplaneDataLoader` (HDF5) + `load_or_compute_norm_stats` |
| `dataset/preprocess_dat_to_h5.py` | Raw Tecplot `.dat` → `.h5`; writes the split json |
| `scripts/transolver_plus.sh` | Single-GPU launch (direct `python`) |
| `airplane_dataset.json` | Train/test split manifest (h5 filenames) |
| `checkpoints/*.pt` | Paper's standard-benchmark weights (reference only; no runner here) |

---

## 4. Data pipeline & format

Raw data (per sample) is a Tecplot FEPOINT surface triangulation:

```
dataset/aircraft_dataset/<id>/Mach<MM.MM>_Alpha<AA.AA>_Beta<BB.BB>/Components.i.dat
VARIABLES = x,y,z, Cp, Rho, U, V, W, Pressure
ZONE ... N=<#nodes>, E=<#tris>, F=FEPOINT, ET=TRIANGLE
```

`dataset/preprocess_dat_to_h5.py` converts each into an `.h5` the model consumes:

| h5 dataset | shape | source |
|------------|-------|--------|
| `pos` | (N, 3) | x, y, z |
| `normals` | (N, 3) | per-node **area-weighted** normals computed from the triangulation |
| `values` | (N, 6) | Cp, Rho, U, V, W, Pressure |
| attrs | — | `Ma`, `alpha`, `beta` |

- **Canonical naming** (single source of truth for both training json and eval):
  `{id}_{Ma}_{alpha}_{beta}.h5`, e.g. `91_7_7_2.h5`. `result.csv` is the master sample index.
- **Normalization** statistics (mean/std of pos/normals/values) are computed once from the
  training set and cached to `<save_dir>/norm_stats.json`. The auto-computed constants reproduce
  the paper's original hardcoded "200-case" values (e.g. `pos_mean[0] ≈ 2808.79`).

---

## 5. Standard workflow (single GPU)

```bash
# (env) use the conda ml environment

# 1) Preprocess raw .dat -> .h5 and write the train/test split json.
python dataset/preprocess_dat_to_h5.py --ids 1 2 3 11 12 91 92 --write_split --test_ids 91 92

# 2) Train (+ validate every --val_iter epochs). norm_stats.json is auto-created.
python main_airplane.py --nb_epochs 2 --val_iter 1 --dataset airplane \
    --cfd_model=transolver_plus \
    --data_dir dataset/aircraft_dataset/ --save_dir dataset/aircraft_dataset/

# 3) Offline evaluation on the test split (loads train/model_<nb_epochs>.pth).
python main_airplane.py --nb_epochs 2 --eval 1 --dataset airplane \
    --cfd_model=transolver_plus \
    --data_dir dataset/aircraft_dataset/ --save_dir dataset/aircraft_dataset/
```

- `scripts/transolver_plus.sh` wraps the single-GPU training invocation.
- All artifacts (logs, `model_<N>.pth`, `log_<N>.json`, `output/*.npy`) are written under `train/`.
- Single GPU skips the process group entirely (`WORLD_SIZE` defaults to 1). For multi-GPU, launch
  with `torchrun` and `WORLD_SIZE>1` (nccl on Linux, gloo elsewhere).

---

## 6. Fixes applied (to make the standard flow run)

The upstream code targeted a multi-GPU Linux/NCCL cluster and shipped without a `.dat→.h5`
preprocessor. The following were fixed:

| Area | Problem | Fix |
|------|---------|-----|
| Distributed | hardcoded `backend="nccl"` (unavailable on Windows); gloo+CUDA `all_reduce` segfaults | skip the process group on a single GPU; guard `all_reduce` to `world_size>1` (`_is_distributed()`); auto-select nccl/gloo for multi-GPU |
| Launcher | deprecated `torch.distributed.launch` + libuv error on Windows | single-GPU runs call `python` directly; `USE_LIBUV=0` set when a group is needed |
| Data | no `.dat→.h5` preprocessing; required normals not present | added `dataset/preprocess_dat_to_h5.py` (parses Tecplot, computes vertex normals) |
| Paths | hardcoded `/aircraft_docker/...json`, `/aircraft_data/`, `./model_200.pth` | manifest resolved from `save_dir`/repo-root/`AIRPLANE_JSON`; eval paths parameterized |
| Normalization | hardcoded "200-case" constants | auto-computed from the training set and cached (`load_or_compute_norm_stats`) |
| Eval branch | magic `df.iloc[-14:]`, filename mismatch | iterate the test split, read `Ma/alpha/beta` from h5, canonical naming, cross-ref `result.csv` |
| Artifacts | dumped at repo root | centralized under `train/` |

---

## 7. Verified test results

Run on RTX 5080 (16 GB), conda `ml`, with a 25-train / 10-test split (5 aircraft × 5 conditions
train; 2 aircraft × 5 conditions test):

| Stage | Outcome |
|-------|---------|
| Preprocess | 35 `.h5` generated, ~0.63 s/sample, ~15 MB each |
| Train (2 epochs) | `train_loss` 1.03 → 0.87, `val_l2re` 0.764 → 0.677, 33 s, all on GPU |
| Full-mesh fwd+bwd | 331,971 points, peak GPU memory **9.88 GB / 16 GB** |
| Eval (10 samples) | mean pressure L2RE **0.6767** — identical to the final training `val_l2re`, confirming the eval and validation paths are consistent |

The L2RE is high only because training was a 2-epoch smoke run; the pipeline itself is correct
and stable end-to-end. The model logic was independently verified on CPU (loss decreases, fwd/bwd OK).

---

## 8. Migrating to a new dataset

The pipeline is dataset-agnostic for AirCraft-style surface data:

1. Place raw folders `<id>/Mach<MM.MM>_Alpha<AA.AA>_Beta<BB.BB>/Components.i.dat` and a
   `result.csv` (columns `idx,Ma,alpha,beta,...`) under `--raw_dir`.
2. Run preprocessing (`--write_split` to regenerate the split, `--test_ids` to choose the test set).
3. Train and eval as in §5. Normalization is recomputed automatically — delete
   `norm_stats.json` to force a refresh.

Adjust `Model(...)` in `main_airplane.py` only if channel counts differ: `space_dim` (input
per-node dims), `out_dim` (predicted fields), and the `condition` dimensionality (currently
`[Ma, alpha, beta]` → 3). These are architecture-config choices, not pipeline issues.

---

## 9. Notes & limitations

- `checkpoints/*.pt` are **standard-benchmark** weights from the paper release; this repo has no
  benchmark data loader/runner, so they cannot be evaluated here without porting that code.
- The eval branch reports the model's **field** prediction error (pressure L2RE) and the reference
  aerodynamic coefficients from `result.csv`. Predicting coefficients (CD/CL) directly would
  require surface integration and is not implemented.
- Generated data and artifacts are gitignored: `dataset/aircraft_dataset/` (raw + `.h5` +
  `norm_stats.json`), `dataset/aircraft_dataset.zip`, and `train/`.
