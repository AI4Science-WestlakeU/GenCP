#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE

export PYTHONPATH=/path/to/project/M2PDE

export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --num_processes 1 \
    --mixed_precision 'fp16' \
    --main_process_port 29500 \
    train/turek_hron.py \
    --model_type CNO \
    --stage fluid \
    --lr 0.0003 \
    --input_size 108,88 \
    --n_layers 2 \
    --channel_multiplier 32 \
    --num_steps 100000 \
    --use_validation False