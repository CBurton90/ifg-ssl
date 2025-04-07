#!/bin/bash

#SBATCH --job-name MAE_linear_train-eval
#SBATCH --time=2:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/MAE_linear_train-eval_hephaestus_224crop.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch
python3 training/MAE_linear_train-eval.py
conda deactivate