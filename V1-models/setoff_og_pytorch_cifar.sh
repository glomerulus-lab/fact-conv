#!/bin/bash                                          
#SBATCH --gres=gpu:a100l:1                                 
#SBATCH --constraint="ampere"
#SBATCH -c 8                                  
#SBATCH --mem=60G                                    
#SBATCH  -t 22:00:00                                 
#SBATCH --output slurm/%j.out                     

module load python/3.8
module load libffi
source newenv/bin/activate

python og_pytorch_cifar.py --freeze_spatial $1 --freeze_channel $2 --spatial_init $3 --net $4 --name $5
