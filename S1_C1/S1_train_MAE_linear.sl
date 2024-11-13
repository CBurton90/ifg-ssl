#!/bin/bash

#SBATCH --job-name MAE_S1_lin_eval
#SBATCH --time=1:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/SynInSAR_MAE_ViTB16_lin_eval.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch
python3 training/MAE_S1_lin_eval.py
conda deactivate