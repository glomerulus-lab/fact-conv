#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=124G
#SBATCH --time=3-00:00:00 
#SBATCH --output=slurm/%j.ou
#SBATCH --nodes=1
#SBATCH --account=rrg-bengioy-ad
module load python/3.8
source env_imagenet/bin/activate
srun --job-name="Dataset_Staging" --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES  --ntasks-per-node=1 bash unpack_imagenet.sh

export MASTER_ADDR=$(hostname -i "$SLURMD_NODENAME")
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=TRACE

exec srun --gpus-per-task 1 --job-name="Main_Run"  python imagenet.py --dist-url env://  --workers  $SLURM_CPUS_PER_TASK --width_scale $SLURM_NTASKS_PER_NODE  "$SLURM_TMPDIR/imagenet" "$@" 
