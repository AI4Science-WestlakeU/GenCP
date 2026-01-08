#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE
export PYTHONPATH=/path/to/project/M2PDE

export NTCOUPLE_DATA_ROOT=/path/to/dataset/NTcouple
export CUDA_VISIBLE_DEVICES=0

FLUID_CKPT=/path/to/results/ntcouple_m2pde_fluid_sit_fno/checkpoint/model.pt

python eval/infer_single_ntcouple.py \
    --stage fluid \
    --dataset decouple_val \
    --checkpoint "${FLUID_CKPT}" \
    --batch-size 50 \
    --diffusion-step 250 \
    --model-type SiT_FNO \
    --patch-size 2,1 \
    --num-frames 16 \
    --hidden-size 256 \
    --depth 4 \
    --num-heads 4 \
    --modes 4


