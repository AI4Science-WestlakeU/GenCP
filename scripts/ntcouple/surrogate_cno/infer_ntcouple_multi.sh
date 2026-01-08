#!/bin/bash
# Inference script for NTcouple multi-field (neutron, solid, fluid) with surrogate models

export CUDA_VISIBLE_DEVICES=0
export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project/GenCP

# Set checkpoint paths for three fields (surrogate models)
# Update these paths to your trained surrogate model checkpoints
NEUTRON_CHECKPOINT=/path/to/results/ntcouple/neutron_CNO_surrogate/checkpoint/best_main_model.pth
SOLID_CHECKPOINT=/path/to/results/ntcouple/solid_CNO_surrogate/checkpoint/best_main_model.pth
FLUID_CHECKPOINT=/path/to/results/ntcouple/fluid_CNO_surrogate/checkpoint/best_main_model.pth

# Evaluate on couple_val (coupled evaluation split)
echo ""
echo "Evaluating NTcouple multi-field inference with surrogate models on couple_val (coupled evaluation)..."
echo "Neutron checkpoint: ${NEUTRON_CHECKPOINT}"
echo "Solid checkpoint: ${SOLID_CHECKPOINT}"
echo "Fluid checkpoint: ${FLUID_CHECKPOINT}"
echo ""

python infer_multi_ntcouple_surrogate.py \
  --config configs/ntcouple/multi_surrogate_cno.yaml \
  --ntcouple-neutron-checkpoint-path ${NEUTRON_CHECKPOINT} \
  --ntcouple-solid-checkpoint-path ${SOLID_CHECKPOINT} \
  --ntcouple-fluid-checkpoint-path ${FLUID_CHECKPOINT} \
  --eval-split couple_val \
  --use-surrogate \
  --num-inference-steps 300 \
  --save-figs-path ./visualization_results/ntcouple_multi_surrogate \
  --seed 42

echo ""
echo "Multi-field surrogate inference completed!"

