#!/bin/bash                                          
#SBATCH --gres=gpu:v100l:2                                 
#SBATCH -c 16                                  
#SBATCH --ntasks-per-node=2 
#SBATCH --mem=92G                                    
#SBATCH --time 72:00:00                                 
#SBATCH --output slurm/%j.out                     
#SBATCH --account=rrg-bengioy-ad

module load python/3.8
source env_imagenet/bin/activate

bash unpack_imagenet.sh

python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:56515 --nnodes=1 --nproc_per_node 2 imagenet.py --multiprocessing-distributed --dist-url env:// --workers 8 --arch $1 --fact $2 --name $3 --width_scale $5 -b $6 $SLURM_TMPDIR/imagenet/ 

#python imagenet.py --arch $1 --fact $2 --name $3 --lr $4 --width_scale $5 --batch_size $6
