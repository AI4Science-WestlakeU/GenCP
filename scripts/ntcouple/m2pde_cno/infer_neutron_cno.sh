#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE
export PYTHONPATH=/path/to/project/M2PDE

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple
export CUDA_VISIBLE_DEVICES=0

NEUTRON_CKPT="/path/to/results/ntcouple_m2pde_neutron_cno/checkpoint/model.pt"

python eval/infer_single_ntcouple.py \
    --stage neutron \
    --dataset couple_val \
    --checkpoint "${NEUTRON_CKPT}" \
    --batch-size 1000 \
    --diffusion-step 250
