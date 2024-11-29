#!/bin/bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --constraint="ampere"
#SBATCH -c 8
#SBATCH --mem=20G
#SBATCH  -t 20:00:00
#SBATCH --output slurm/%j.out
#SBATCH --partition long

module load anaconda/3
conda activate random_features

python pytorch_cifar.py --nonlinearity "abs" --width $1 --channel_k $2 --seed $3 --net $4
