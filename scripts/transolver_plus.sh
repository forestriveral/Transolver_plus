export CUDA_VISIBLE_DEVICES=0

# Single-GPU run: no distributed launcher needed. main_airplane.py defaults to
# WORLD_SIZE=1 and skips process-group init, so all compute stays on the GPU.
# For multi-GPU, launch with torchrun and WORLD_SIZE>1 (nccl on Linux, gloo elsewhere).
#
# Training + validation (the standard flow): omit --eval.
# Offline evaluation: pass --eval 1 (requires a trained model_<nb_epochs>.pth and
# a result.csv under --save_dir).

DATA_DIR=${DATA_DIR:-dataset/aircraft_dataset/}

python main_airplane.py \
    --nb_epochs 200 \
    --fold_id 0 \
    --dataset airplane \
    --cfd_model=transolver_plus \
    --data_dir "$DATA_DIR" \
    --save_dir "$DATA_DIR"
