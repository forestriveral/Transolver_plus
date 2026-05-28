# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Transolver++ (ICML 2025) — an accurate neural PDE solver for million-scale geometries.
Theory lives in `references/2502.02414v2.pdf`; this repo is the official implementation.
Forked from `thuml/Transolver_plus` (set as `upstream`); `origin` is the personal fork.

## Key files

- `models/Transolver_plus.py` — the model. `Physics_Attention_1D_Eidetic` is the core block.
- `main_airplane.py` — entry point: arg parsing, DDP init, model construction, train/eval dispatch.
- `train_airplane.py` — training loop (`train.main(...)`).
- `dataset/dataset.py` — `AirplaneDataset` / `AirplaneDataLoader`, read HDF5 samples.
- `scripts/transolver_plus.sh` — launch script (`torch.distributed.launch`).

## Running

`bash scripts/transolver_plus.sh`. The model **always** runs under DDP: `main_airplane.py`
calls `dist.init_process_group(backend="nccl", ...)` and the attention does `dist_nn.all_reduce`,
so it must be launched via `torch.distributed.launch` even on a single GPU.

## Gotchas (read before running locally)

- **Hardcoded absolute paths** that must be edited for any local run:
  - `dataset/dataset.py:11` → `/aircraft_docker/airplane_dataset.json`
  - `main_airplane.py` eval branch (~L105-106) → `/aircraft_data/`, `result.csv`, and `./model_200.pth`
- **Data format mismatch**: code consumes `.h5` files (datasets `pos` / `normals` / `values`, attrs
  `Ma` / `alpha` / `beta`). The shipped data under `dataset/aircraft_dataset/` is raw
  `Components.i.dat`. There is **no `.dat → .h5` preprocessing script in the repo** — it is a known gap.
- `--eval` uses argparse `type=bool`, so any non-empty string is truthy (the script passes `--eval 1`).
- Large data is gitignored: `dataset/aircraft_dataset.zip` (~2.5GB) and `dataset/aircraft_dataset/`.

## Paper ↔ code mapping (for model edits)

- Eidetic states = `gumbel_softmax()` (Rep-Slice, Eq. 4) + `proj_temperature` (Ada-Temp, Eq. 3),
  replacing the original Softmax slice weights.
- Parallel framework = `dist_nn.all_reduce` on `slice_norm` and `slice_token`
  (`Transolver_plus.py:71,73`), i.e. Algorithm 1's reduction.
- Memory saving = only `x` is projected (no `f` branch), per the paper's "Further speedup".
