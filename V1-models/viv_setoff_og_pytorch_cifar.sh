#!/bin/bash                                          
#SBATCH --gres=gpu:a100l:1                                 
#SBATCH --constraint="ampere"
#SBATCH -c 8                                  
#SBATCH --mem=60G                                    
#SBATCH  -t 22:00:00                                 
#SBATCH --output slurm/%j.out                     

module load anaconda/3
conda activate random_features

#python resnet_og_pytorch_cifar.py --net $1 --name $2 --width_scale $3
python width_8_pytorch_cifar.py --net $1 --name $2 --width_scale $3
