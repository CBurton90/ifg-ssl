#!/bin/bash

#SBATCH --job-name MAE_pretraining
#SBATCH --time=2-00:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/MAE_pretrain_hephaestus_v2.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch
python3 training/MAE_pretrain.py
conda deactivate