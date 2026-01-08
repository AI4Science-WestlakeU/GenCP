#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE

export PYTHONPATH=/path/to/project/M2PDE

export CUDA_VISIBLE_DEVICES=0
export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple

accelerate launch --num_processes 1 --mixed_precision fp16 --main_process_port 29500 \
    train/ntcouple.py \
    --stage fluid \
    --dataset decouple_train \
    --n-dataset 4000 \
    --diffusion-step 250 \
    --lr 5e-4 \
    --batchsize 16 \
    --gradient-accumulate-every 1 \
    --model-type SiT_FNO \
    --patch-size 2,1 \
    --num-frames 16 \
    --hidden-size 256 \
    --depth 4 \
    --num-heads 4 \
    --modes 4 \
    --exp-id ntcouple_m2pde_fluid_sit_fno \
    --checkpoint 5000
