#!/bin/bash

sbatch --ntasks-per-node 1 setoff_${CC_CLUSTER}.sh --arch resnet18 --fact True --name fact_resnet_width_1 --lr 0.1 --width_scale 1 -b 1024
sbatch --ntasks-per-node 1 setoff_${CC_CLUSTER}.sh --arch resnet18 --fact False --name conv_resnet_width_1 --lr 0.1 --width_scale 1 -b 1024

sbatch --ntasks-per-node 1 setoff_${CC_CLUSTER}.sh --arch alexnet --fact True --name fact_alexnet_width_1 --lr 0.01 --width_scale 1 -b 1024
sbatch --ntasks-per-node 1 setoff_${CC_CLUSTER}.sh --arch alexnet --fact False --name conv_alexnet_width_1 --lr 0.01 --width_scale 1 -b 1024

sbatch --ntasks-per-node 2 setoff_${CC_CLUSTER}.sh --arch resnet18 --fact True --name fact_resnet_width_2 --lr 0.1 --width_scale 2 -b 512

sbatch --ntasks-per-node 4 setoff_${CC_CLUSTER}.sh --arch resnet18 --fact True --name fact_resnet_width_4 --lr 0.1 --width_scale 4 -b 256
