#!/bin/bash

#SBATCH --job-name linear_S1_train
#SBATCH --time=1:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/linear_ViTS16_S1_train.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch
pwd
python3 training/linear_S1_train.py
conda deactivate
