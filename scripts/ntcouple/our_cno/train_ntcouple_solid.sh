#!/bin/bash
# Training script for NTcouple solid field with GenCP framework (CNO backbone)

source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project/GenCP

# Single GPU training (using GPU 0)
export CUDA_VISIBLE_DEVICES=0
python train.py \
  --config configs/ntcouple/solid_cno.yaml
