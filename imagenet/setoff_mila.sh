#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=132G
#SBATCH --output=slurm/%j.out
#SBATCH --nodes=1
#SBATCH --constraint="ampere"
#SBATCH --partition=short-unkillable

module load python/3.8
source env_imagenet/bin/activate

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=TRACE
#export CUDA_LAUNCH_BLOCKING=1
#exec srun --job-name="Main_Run"  python imagenet.py --dist-url env://  --workers  $SLURM_CPUS_PER_TASK "/network/datasets/imagenet.var/imagenet_torchvision/" "$@"

exec srun --job-name="Main_Run" debug-wrap.sh imagenet.py --dist-url env:// --workers $SLURM_CPUS_PER_TASK "/network/datasets/imagenet.var/imagenet_torchvision/" "$@"
