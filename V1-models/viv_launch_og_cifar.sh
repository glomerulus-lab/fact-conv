#!/bin/bash

#sbatch viv_setoff_og_pytorch_cifar.sh vgg compare_vgg
#sbatch viv_setoff_og_pytorch_cifar.sh vggbn compare_vggbn
#sbatch viv_setoff_og_pytorch_cifar.sh resnet compare_resnet

#sbatch viv_setoff_og_pytorch_cifar.sh resnet 0.125scale 0.125
#sbatch viv_setoff_og_pytorch_cifar.sh resnet 0.25scale 0.25
#sbatch viv_setoff_og_pytorch_cifar.sh resnet 0.5scale 0.5
#sbatch viv_setoff_og_pytorch_cifar.sh resnet 1scale 1
#sbatch viv_setoff_og_pytorch_cifar.sh resnet 2scale 2
#sbatch viv_setoff_og_pytorch_cifar.sh resnet 4scale 4

#sbatch viv_setoff_k_pytorch_cifar.sh resnet 1k 1
#sbatch viv_setoff_k_pytorch_cifar.sh resnet 2k 2
#sbatch viv_setoff_k_pytorch_cifar.sh resnet 4k 4
#sbatch viv_setoff_k_pytorch_cifar.sh resnet 8k 8
#sbatch viv_setoff_k_pytorch_cifar.sh resnet 16k 16

#sbatch three_layers_setoff.sh ThreeLayerModels
sbatch viv_setoff_og_pytorch_cifar.sh resnet 8scale 8
