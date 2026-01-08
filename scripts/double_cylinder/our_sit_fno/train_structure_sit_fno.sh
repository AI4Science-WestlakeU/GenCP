#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0
accelerate launch \
    --num_processes 1 \
    --mixed_precision 'no' \
    train.py --config configs/double_cylinder/structure_sit_fno.yaml \
    --use-accelerator \
    --log-every 1000 \
    --ckpt-every 5000
