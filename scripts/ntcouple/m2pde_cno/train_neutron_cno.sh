#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE

export PYTHONPATH=/path/to/project/M2PDE

export CUDA_VISIBLE_DEVICES=0
export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple

accelerate launch --num_processes 1 --mixed_precision fp16 --main_process_port 29500 \
    train/ntcouple.py \
    --stage neutron \
    --dataset decouple_train \
    --n-dataset 4000 \
    --diffusion-step 250 \
    --lr 5e-4 \
    --batchsize 16 \
    --gradient-accumulate-every 1 \
    --model-type CNO \
    --n-layers 2 \
    --channel-multiplier 16 \
    --cno-input-size 64 \
    --exp-id ntcouple_m2pde_neutron_cno \
    --checkpoint 5000
