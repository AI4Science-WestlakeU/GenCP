#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gencp

cd /path/to/project/GenCP
export PYTHONPATH=/path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0
python infer_single.py --config configs/turek_hron/structure_surrogate_cno.yaml
