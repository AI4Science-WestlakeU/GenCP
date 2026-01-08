#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE

export PYTHONPATH=/path/to/project/M2PDE

export CUDA_VISIBLE_DEVICES=0

python eval/infer_single_turek_hron.py \
    --checkpoint_path /path/to/results/turek_hron/diffusionCNOfluid/checkpoint/model.pt \
    --stage fluid \
    --model_name CNO \
    --test_batch_size 8 \
    --length 999 \
    --input_step 3 \
    --output_step 12 \
    --input_size 108,88 \
    --n_layers 2 \
    --channel_multiplier 32 \
    --stride 1 \
    --num_delta_t 0 \
    --dt 5 \
    --sdf_threshold 0.04 \
    --ddim_sampling_timesteps 250 \
    --diffusion_step 250 \
    --num_workers 16 \
    --seed 42
