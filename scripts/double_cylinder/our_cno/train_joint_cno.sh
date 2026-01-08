#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp
cd /path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
    --num_processes 2 \
    --mixed_precision 'no' \
    --main_process_port 29500 \
    train.py --config configs/double_cylinder/joint_cno.yaml \
    --use-accelerator \
    --log-every 1000 \
    --ckpt-every 5000 
