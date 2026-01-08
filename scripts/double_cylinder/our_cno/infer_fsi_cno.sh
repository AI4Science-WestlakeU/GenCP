#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh    
conda activate gencp
cd /path/to/project/GenCP
export PYTHONPATH=/path/to/project/GenCP

export CUDA_VISIBLE_DEVICES=0
python infer_multi.py --config configs/double_cylinder/fsi_cno.yaml
