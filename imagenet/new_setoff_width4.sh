#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=v100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=184G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/%j.out
#SBATCH --account=rrg-bengioy-ad

module load python/3.8
source env_imagenet/bin/activate

bash unpack_imagenet.sh

exec python -m torch.distributed.run         \
    --rdzv-backend=c10d                      \
    --rdzv-endpoint=localhost:0              \
    --nnodes=$SLURM_NNODES                   \
    --nproc-per-node=$SLURM_NTASKS_PER_NODE  \
    imagenet.py --multiprocessing-distributed         \
                --dist-url    env://                  \
                --workers     $SLURM_CPUS_PER_TASK    \
                --width_scale $SLURM_NTASKS_PER_NODE  \
                "$SLURM_TMPDIR/imagenet" "$@"
