#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=46G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --output=slurm/%j.out
#SBATCH --account=rrg-bengioy-ad
exec ./setoff.sh "$@"
