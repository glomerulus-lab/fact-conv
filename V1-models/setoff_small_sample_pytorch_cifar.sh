#!/bin/bash                                          
#SBATCH --gres=gpu:a100l:1                                 
#SBATCH --constraint="ampere"
#SBATCH -c 8                                  
#SBATCH --mem=60G                                    
#SBATCH  -t 08:00:00                                 
#SBATCH --output slurm/%j.out                     

module load python/3.8
source ../env/bin/activate

python small_sample_pytorch_cifar.py --freeze_spatial $1 --freeze_channel $2 --spatial_init $3 --name $4
