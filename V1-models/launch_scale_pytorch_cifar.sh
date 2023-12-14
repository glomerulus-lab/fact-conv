#!/bin/bash
for i in 1 2 4 8 16
do
  sbatch setoff_pytorch_cifar_viv.sh False False V1 Pytorch_Cifar_Resnet_V1_LSLC $i 
  sbatch setoff_pytorch_cifar_viv.sh False False default Pytorch_Cifar_Resnet_default_LSLC $i

  sbatch setoff_pytorch_cifar_viv.sh True True V1 Pytorch_Cifar_Resnet_V1_USUC $i 
  sbatch setoff_pytorch_cifar_viv.sh True True default Pytorch_Cifar_Resnet_default_USUC $i

  sbatch setoff_pytorch_cifar_viv.sh True False V1 Pytorch_Cifar_Resnet_V1_USLC $i 
  sbatch setoff_pytorch_cifar_viv.sh True False default Pytorch_Cifar_Resnet_default_USLC $i

  sbatch setoff_pytorch_cifar_viv.sh False True V1 Pytorch_Cifar_Resnet_V1_LSUC $i 
  sbatch setoff_pytorch_cifar_viv.sh False True default Pytorch_Cifar_Resnet_default_LSUC $i
done


#for i in {0..9}
#do
#  sbatch setoff_pytorch_cifar100.sh False False V1 Pytorch_Cifar100_Resnet_V1_LSLC $i 
#  sbatch setoff_pytorch_cifar100.sh False False default Pytorch_Cifar100_Resnet_default_LSLC $i
#
#  sbatch setoff_pytorch_cifar100.sh True True V1 Pytorch_Cifar100_Resnet_V1_USUC $i 
#  sbatch setoff_pytorch_cifar100.sh True True default Pytorch_Cifar100_Resnet_default_USUC $i
#
#  sbatch setoff_pytorch_cifar100.sh True False V1 Pytorch_Cifar100_Resnet_V1_USLC $i 
#  sbatch setoff_pytorch_cifar100.sh True False default Pytorch_Cifar100_Resnet_default_USLC $i
#
#  sbatch setoff_pytorch_cifar100.sh False True V1 Pytorch_Cifar100_Resnet_V1_LSUC $i 
#  sbatch setoff_pytorch_cifar100.sh False True default Pytorch_Cifar100_Resnet_default_LSUC $i
#done
#

#sbatch setoff_pytorch_cifar_viv.sh False False V1 Pytorch_Cifar_Resnet_V1_LSLC
#sbatch setoff_pytorch_cifar_viv.sh True False V1 Pytorch_Cifar_Resnet_V1_USLC
#sbatch setoff_pytorch_cifar_viv.sh False True V1 Pytorch_Cifar_Resnet_V1_LSUC
#sbatch setoff_pytorch_cifar_viv.sh True True V1 Pytorch_Cifar_Resnet_V1_USUC


#sbatch setoff_pytorch_cifar_viv.sh False False default Pytorch_Cifar_Resnet_default_LSLC
#sbatch setoff_pytorch_cifar_viv.sh True False default Pytorch_Cifar_Resnet_default_USLC
#sbatch setoff_pytorch_cifar_viv.sh False True default Pytorch_Cifar_Resnet_default_LSUC
#sbatch setoff_pytorch_cifar_viv.sh True True default Pytorch_Cifar_Resnet_default_USUC

#sbatch setoff_pytorch_cifar_viv.sh False False V1 Pytorch_Cifar_Resnet_V1_LSLC_freeze_test
#sbatch setoff_pytorch_cifar_viv.sh True False V1 Pytorch_Cifar_Resnet_V1_USLC_freeze_test
#sbatch setoff_pytorch_cifar_viv.sh False True V1 Pytorch_Cifar_Resnet_V1_LSUC_freeze_test
#sbatch setoff_pytorch_cifar_viv.sh True True V1 Pytorch_Cifar_Resnet_V1_USUC_freeze_test
#sbatch setoff_pytorch_cifar_viv.sh False False V1 post New_Pytorch_Cifar_Resnet_V1_post_slurm
#sbatch setoff_pytorch_cifar_viv.sh False False V1 pre New_Pytorch_Cifar_Resnet_V1_pre_slurm
#sbatch setoff_pytorch_cifar_viv.sh True False V1 post New_Pytorch_Cifar_Resnet_V1_USLC_post_slurm
#sbatch setoff_pytorch_cifar_viv.sh True False V1 pre New_Pytorch_Cifar_Resnet_V1_USLC_pre_slurm
#sbatch setoff_og_pytorch_cifar.sh False False V1 pre New_Pytorch_Cifar_Resnet
#bash setoff_pytorch_cifar_viv.sh False False V1 post Pytorch_Cifar_Resnet_V1_post | tee post_LCLS_cifar.out
#bash setoff_pytorch_cifar_viv.sh False False V1 pre Pytorch_Cifar_Resnet_V1_pre | tee pre_LCLS_cifar.out
#bash setoff_grad.sh False False V1 post Record_Resnet_V1_frozen_spatial_post | tee record_grad_LCLS_post.out
#bash setoff_grad.sh False False V1 pre Record_Resnet_V1_frozen_spatial_pre | tee record_grad_LCLS_pre.out
