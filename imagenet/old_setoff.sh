#!/bin/bash                                          
#SBATCH --gres=gpu:v100l:1                                 
#SBATCH --cpus-per-task=8                                  
#SBATCH --mem=46G                                    
#SBATCH --time 72:00:00                                 
#SBATCH --output slurm/%j.out                     
#SBATCH --account=rrg-bengioy-ad

module load python/3.8
source env_imagenet/bin/activate

bash unpack_imagenet.sh

python imagenet.py --arch $1 --fact $2 --name $3 --lr $4 --width_scale $5 --batch_size $6 $SLURM_TMPDIR/imagenet/
