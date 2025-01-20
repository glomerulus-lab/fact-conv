#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --output=slurm/%j.out
#SBATCH --nodes=1
#SBATCH --C=dgx
#SBATCH --partition=long

module load python/3.8
source env_imagenet/bin/activate
#srun --job-name="Dataset_Staging" --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES  --ntasks-per-node=1 bash unpack_imagenet.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=TRACE
#--width_scale $SLURM_NTASKS_PER_NODE
exec srun --gpus-per-task 1 --job-name="Main_Run"  python imagenet_vivian.py --dist-url env://  --workers  $SLURM_CPUS_PER_TASK  "/network/datasets/imagenet.var/imagenet_torchvision/" "$@" 
