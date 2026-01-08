#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/GenCP
export PYTHONPATH=/path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0
accelerate launch \
    --num_processes 1 \
    --mixed_precision 'no' \
    train.py --config configs/turek_hron/structure_cno.yaml \
    --use-accelerator \
    --log-every 1000 \
    --ckpt-every 5000
