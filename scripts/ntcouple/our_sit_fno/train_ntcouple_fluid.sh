#!/bin/bash
# Training script for NTcouple fluid field with GenCP framework (SiT_FNO backbone)

source /opt/conda/etc/profile.d/conda.sh
conda activate fsi

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project

# 单卡训练（使用 GPU 2）
export CUDA_VISIBLE_DEVICES=2
python train.py \
  --config configs/ntcouple/fluid_sit_fno.yaml

