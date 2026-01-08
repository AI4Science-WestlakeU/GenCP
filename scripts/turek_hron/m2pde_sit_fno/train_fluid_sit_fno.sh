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
    --model_type SiT_FNO \
    --stage fluid \
    --lr 0.0005 \
    --input_size 108,88 \
    --depth 4 \
    --hidden_size 256 \
    --patch_size 2,2 \
    --num_heads 4 \
    --modes 4 \
    --num_steps 100000 \
    --use_validation False