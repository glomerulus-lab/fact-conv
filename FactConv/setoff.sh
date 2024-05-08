#!/bin/bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --constraint="ampere"
#SBATCH -c 8
#SBATCH --mem=20G
#SBATCH  -t 20:00:00
#SBATCH --output slurm/%j.out
#SBATCH --partition long

module load python/3.8
source ../refactor/env/bin/activate
echo "$@"
CUDA_VISIBLE_DEVICES=0 python pytorch_cifar.py "$@"
