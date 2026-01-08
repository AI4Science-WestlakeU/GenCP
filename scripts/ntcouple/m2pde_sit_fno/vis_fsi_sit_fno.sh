#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE
export PYTHONPATH=/path/to/project/M2PDE

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple
export CUDA_VISIBLE_DEVICES=0

NEUTRON_CKPT=/path/to/results/ntcouple_m2pde_neutron_sit_fno/checkpoint/model.pt
SOLID_CKPT=/path/to/results/ntcouple_m2pde_solid_sit_fno/checkpoint/model.pt
FLUID_CKPT=/path/to/results/ntcouple_m2pde_fluid_sit_fno/checkpoint/model.pt

python eval/infer_multi_ntcouple.py \
    --eval-split couple \
    --checkpoint-neutron "${NEUTRON_CKPT}" \
    --checkpoint-solid "${SOLID_CKPT}" \
    --checkpoint-fluid "${FLUID_CKPT}" \
    --diffusion-step 250 \
    --outer-iters 2 \
    --model-type SiT_FNO \
    --patch-size 2,2 \
    --num-frames 16 \
    --hidden-size 256 \
    --depth 4 \
    --num-heads 4 \
    --modes 4 \
    --batch-size 2500 \
    --viz-samples 5


