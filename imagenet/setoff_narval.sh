#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=124G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --nodes=1
#SBATCH --account=rrg-bengioy-ad
exec setoff.sh "$@"
