#!/bin/bash
# Training script for NTcouple neutron field with surrogate model (SiT_FNO)

source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple/

cd /path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0
python train.py \
  --config configs/ntcouple/neutron_surrogate_sit_fno.yaml
