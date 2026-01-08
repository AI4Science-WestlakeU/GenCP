#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/M2PDE

export PYTHONPATH=/path/to/project/M2PDE

export CUDA_VISIBLE_DEVICES=0

python eval/infer_multi_double_cylinder.py \
    --fluid_checkpoint_path /path/to/results/double_cylinder/diffusionSiT_FNOfluid/checkpoint/model.pt \
    --structure_checkpoint_path /path/to/results/double_cylinder/diffusionSiT_FNOstructure/checkpoint/model.pt \
    --model_name SiT_FNO \
    --test_batch_size 10 \
    --num_iter 2 \
    --length 999 \
    --input_step 3 \
    --output_step 12 \
    --input_size 128,128 \
    --stride 1 \
    --num_delta_t 0 \
    --dt 10 \
    --sdf_threshold 0.04 \
    --ddim_sampling_timesteps 10 \
    --num_workers 16 \
    --seed 42
