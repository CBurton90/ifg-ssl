#!/bin/bash

#SBATCH --job-name dino_train
#SBATCH --time=16:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o hephaestus_train.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch
cd git/ifg-ssl/
python3 hephaestus_dino_gpu.py
conda deactivate
