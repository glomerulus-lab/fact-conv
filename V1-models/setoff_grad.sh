#!/bin/bash                                          
#SBATCH --gres=gpu:1                                 
#SBATCH --constraint="32gb|40gb|48gb"
#SBATCH -c 8                                  
#SBATCH --mem=20G                                    
#SBATCH  -t 06:00:00                                 
#SBATCH --output slurm/%j.out                     

module load python/3.8
module load libffi
source ../env/bin/activate

python Record_Grad_Resnet_V1_CIFAR10.py --freeze_spatial $1 --freeze_channel $2 --spatial_init $3 --net $4 --name $5
