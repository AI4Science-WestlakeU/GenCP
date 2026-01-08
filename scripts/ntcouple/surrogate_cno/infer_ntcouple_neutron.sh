#!/bin/bash
# Inference script for NTcouple neutron field with surrogate model

export CUDA_VISIBLE_DEVICES=0
export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project/GenCP

# Set checkpoint path (update this to your trained surrogate model)
CHECKPOINT_PATH=/path/to/results/ntcouple/neutron_CNO_surrogate/checkpoint/best_main_model.pth

# Evaluate on couple_val (coupled evaluation split)
python infer_single_ntcouple.py \
  --config configs/ntcouple/neutron_surrogate_cno.yaml \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --split decouple_val \
  --num-sampling-steps 100 \
  --max_eval_batches 50 \
  --save_figs_path ./visualization_results/neutron_surrogate \
  --seed 42

