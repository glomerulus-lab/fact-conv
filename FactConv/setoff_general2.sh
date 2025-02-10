#!/bin/bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --constraint="ampere"
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH  -t 12:00:00
#SBATCH --output slurm/%j.out
#SBATCH --partition main

#module load python/3.8
#source ../refactor/env/bin/activate
module load anaconda/3
conda activate random_features

echo "$@"
CUDA_VISIBLE_DEVICES=0 python "$@"
