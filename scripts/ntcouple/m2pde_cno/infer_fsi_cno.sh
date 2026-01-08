#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE
export PYTHONPATH=/path/to/project/M2PDE

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple
export CUDA_VISIBLE_DEVICES=0

FLUID_CKPT="/path/to/results/ntcouple_m2pde_fluid_cno/checkpoint/model.pt"
NEUTRON_CKPT="/path/to/results/ntcouple_m2pde_neutron_cno/checkpoint/model.pt"
SOLID_CKPT="/path/to/results/ntcouple_m2pde_solid_cno/checkpoint/model.pt"

python eval/infer_multi_ntcouple.py \
    --eval-split couple_val \
    --batch-size 50 \
    --checkpoint-neutron "${NEUTRON_CKPT}" \
    --checkpoint-solid "${SOLID_CKPT}" \
    --checkpoint-fluid "${FLUID_CKPT}" \
    --diffusion-step 250 \
    --outer-iters 2 \
    --n-samples 50
