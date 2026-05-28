# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Transolver++ (ICML 2025) — an accurate neural PDE solver for million-scale geometries.
Theory lives in `references/2502.02414v2.pdf`; this repo is the official implementation.
Forked from `thuml/Transolver_plus` (set as `upstream`); `origin` is the personal fork.

## Key files

- `models/Transolver_plus.py` — the model. `Physics_Attention_1D_Eidetic` is the core block.
- `main_airplane.py` — entry point: arg parsing, optional dist init, normalization, train/eval dispatch.
- `train_airplane.py` — training loop + validation (`train.main(...)`, `train.test(...)`).
- `dataset/dataset.py` — `AirplaneDataset` / `AirplaneDataLoader` (HDF5) + `load_or_compute_norm_stats`.
- `dataset/preprocess_dat_to_h5.py` — converts raw Tecplot `.dat` → `.h5`, writes the split json.
- `scripts/transolver_plus.sh` — single-GPU launch (direct `python`, no distributed launcher).

## Running (single GPU)

The AirCraft data ships as raw Tecplot `Components.i.dat`; the model consumes `.h5`. Pipeline:

```bash
# 1. preprocess raw .dat -> .h5 (+ write airplane_dataset.json split). Driven by result.csv.
python dataset/preprocess_dat_to_h5.py --ids 1 2 3 11 12 91 92 --write_split --test_ids 91 92
# 2. train (+validate). Normalization stats auto-computed & cached to <save_dir>/norm_stats.json.
python main_airplane.py --nb_epochs 2 --val_iter 1 --dataset airplane --cfd_model=transolver_plus \
    --data_dir dataset/aircraft_dataset/ --save_dir dataset/aircraft_dataset/
# 3. offline eval on the test split (needs model_<nb_epochs>.pth from step 2).
python main_airplane.py --nb_epochs 2 --eval 1 --dataset airplane --cfd_model=transolver_plus \
    --data_dir dataset/aircraft_dataset/ --save_dir dataset/aircraft_dataset/
```

Single GPU skips the process group entirely (`WORLD_SIZE` defaults to 1); compute stays on GPU.
Multi-GPU: launch with `torchrun` and `WORLD_SIZE>1` (auto-selects nccl on Linux, gloo elsewhere).

## Migrating to a new dataset

The pipeline is dataset-agnostic: provide raw folders `<id>/Mach<MM.MM>_Alpha<AA.AA>_Beta<BB.BB>/Components.i.dat`
plus a `result.csv` (cols `idx,Ma,alpha,beta,...`) under `--raw_dir`, then rerun the 3 steps above.
Normalization is recomputed automatically (delete `norm_stats.json` to force a refresh).

## Gotchas

- **Windows / no-NCCL**: the eidetic-state `all_reduce` is guarded to only run for `world_size>1`
  (`_is_distributed()` in `models/Transolver_plus.py`); gloo+CUDA all_reduce segfaults, so never
  init a process group for a single GPU.
- `--eval` uses argparse `type=bool`, so any non-empty string is truthy — omit it to train.
- Training writes `model_<N>.pth`, `*.log`, `log_<N>.json` to the repo root (`path="./"`); all gitignored.
- Large/derived data is gitignored: `dataset/aircraft_dataset.zip`, `dataset/aircraft_dataset/` (raw +
  generated `.h5` + `norm_stats.json`), `output/`, `*.pth`.
- `checkpoints/*.pt` are the paper's **standard-benchmark** weights (airfoil/darcy/elas/pipe/plas) —
  there is no benchmark runner in this repo (AirCraft code only), so they are reference artifacts.

## Paper ↔ code mapping (for model edits)

- Eidetic states = `gumbel_softmax()` (Rep-Slice, Eq. 4) + `proj_temperature` (Ada-Temp, Eq. 3),
  replacing the original Softmax slice weights.
- Parallel framework = guarded `dist_nn.all_reduce` on `slice_norm` and `slice_token`, i.e. Algorithm 1.
- Memory saving = only `x` is projected (no `f` branch), per the paper's "Further speedup".
