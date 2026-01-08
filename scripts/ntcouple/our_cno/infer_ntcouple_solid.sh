#!/bin/bash
# Inference script for NTcouple solid/fuel field with GenCP framework (CNO backbone)

export CUDA_VISIBLE_DEVICES=0
export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project/GenCP

# Set checkpoint path (update this to your trained model)
CHECKPOINT_PATH="/path/to/results/ntcouple/solid_CNO/ckpt/best_main_model.pth"

# Evaluate on couple_val (coupled evaluation split)
python infer_single_ntcouple.py \
  --config configs/ntcouple/solid_cno.yaml \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --split couple_val \
  --num-sampling-steps 100 \
  --max_eval_batches 1 \
  --save_figs_path ./visualization_results/solid \
  --seed 42
