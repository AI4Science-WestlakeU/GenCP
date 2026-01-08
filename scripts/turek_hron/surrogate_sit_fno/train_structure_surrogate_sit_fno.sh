#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/GenCP
export PYTHONPATH=/path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0
accelerate launch \
    --num_processes 1 \
    --mixed_precision 'fp16' \
    --main_process_port 29500 \
    train.py --config configs/turek_hron/structure_surrogate_sit_fno.yaml \
    --use-accelerator \
    --log-every 100 \
    --ckpt-every 1000
