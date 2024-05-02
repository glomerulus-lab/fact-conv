#!/bin/bash                                          

#SBATCH --gres=gpu:a100l:1                                 
#SBATCH --constraint="ampere"
#SBATCH -c 8                                  
#SBATCH --mem=60G                                    
#SBATCH  -t 08:00:00                                 
#SBATCH --output slurm/%j.out                     

module load anaconda/3
conda activate random_features

python pytorch_cifar_viv.py --freeze_spatial $1 --freeze_channel $2 --spatial_init $3 --name $4 --param_scale $5
