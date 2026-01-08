#!/bin/bash
# Inference script for NTcouple multi-field (neutron, solid, fluid) with GenCP framework (SiT_FNO backbone)

export CUDA_VISIBLE_DEVICES=0
export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project/GenCP

NEUTRON_CHECKPOINT="/path/to/results/ntcouple/neutron_SiT_FNO/ckpt/best_main_model.pth"
SOLID_CHECKPOINT="/path/to/results/ntcouple/solid_SiT_FNO/ckpt/best_main_model.pth"
FLUID_CHECKPOINT="/path/to/results/ntcouple/fluid_SiT_FNO/ckpt/best_main_model.pth"

# Evaluate on couple_val (coupled evaluation split)
echo ""
echo "Evaluating NTcouple multi-field inference on couple_val (coupled evaluation) with SiT_FNO..."
echo "Neutron checkpoint: ${NEUTRON_CHECKPOINT}"
echo "Solid checkpoint: ${SOLID_CHECKPOINT}"
echo "Fluid checkpoint: ${FLUID_CHECKPOINT}"
echo ""

python infer_multi_ntcouple.py \
  --config configs/ntcouple/multi_sit_fno.yaml \
  --ntcouple-neutron-checkpoint-path ${NEUTRON_CHECKPOINT} \
  --ntcouple-solid-checkpoint-path ${SOLID_CHECKPOINT} \
  --ntcouple-fluid-checkpoint-path ${FLUID_CHECKPOINT} \
  --eval-split couple_val \
  --num-sampling-steps 100 \
  --seed 42

echo ""
