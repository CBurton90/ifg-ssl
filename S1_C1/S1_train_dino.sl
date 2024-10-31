#!/bin/bash

#SBATCH --job-name dino_S1_train
#SBATCH --time=14:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/dino_ResNet50_S1_train_no_oversampling.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch
pwd
python3 training/dino_S1_train.py
conda deactivate
