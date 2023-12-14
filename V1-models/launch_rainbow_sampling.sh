#!/bin/bash
module load python/3.8
source newenv/bin/activate

a=(True False)
b=(True False)
c=(rainbow random)
for i in ${a[@]}
do
  for j in ${b[@]}
  do
    for k in ${c[@]}
    do
      python rainbow_network.py --bn_on $i --fact $j --sampling $k
    done
  done
done

#python og_pytorch_cifar.py --net vgg --name vgg | tee results_rn_vgg.out
#python og_pytorch_cifar.py --net vggbn --name vggbn | tee results_rn_vggbn.out
#python og_pytorch_cifar.py --net resnet --name resnet | tee results_rn_resnet.out
#python og_pytorch_cifar.py --net factnetv1 --name factnetv1 | tee results_rn_factnetv1.out
#python og_pytorch_cifar.py --net factnetdefault --name factnetdefault | tee results_rn_factnetdefault.out

#python og_pytorch_cifar.py --net vggfact --name vggfact | tee results_rn_vggfact.out
#python og_pytorch_cifar.py --net vggbnfact --name vggbnfact | tee results_rn_vggbnfact.out
#python og_pytorch_cifar.py --net vgg --name vgg_redo | tee results_rn_vgg_redo.out
#python og_pytorch_cifar.py --net vggbn --name vggbn_redo | tee results_rn_vggbn_redo.out

#sbatch setoff_pytorch_cifar.sh False False V1 Pytorch_Cifar_Resnet_V1_LSLC
#sbatch setoff_pytorch_cifar.sh True False V1 Pytorch_Cifar_Resnet_V1_USLC
#sbatch setoff_pytorch_cifar.sh False True V1 Pytorch_Cifar_Resnet_V1_LSUC
#sbatch setoff_pytorch_cifar.sh True True V1 Pytorch_Cifar_Resnet_V1_USUC


#sbatch setoff_pytorch_cifar.sh False False default Pytorch_Cifar_Resnet_default_LSLC
#sbatch setoff_pytorch_cifar.sh True False default Pytorch_Cifar_Resnet_default_USLC
#sbatch setoff_pytorch_cifar.sh False True default Pytorch_Cifar_Resnet_default_LSUC
#sbatch setoff_pytorch_cifar.sh True True default Pytorch_Cifar_Resnet_default_USUC

#sbatch setoff_pytorch_cifar.sh False False V1 Pytorch_Cifar_Resnet_V1_LSLC_freeze_test
#sbatch setoff_pytorch_cifar.sh True False V1 Pytorch_Cifar_Resnet_V1_USLC_freeze_test
#sbatch setoff_pytorch_cifar.sh False True V1 Pytorch_Cifar_Resnet_V1_LSUC_freeze_test
#sbatch setoff_pytorch_cifar.sh True True V1 Pytorch_Cifar_Resnet_V1_USUC_freeze_test
#sbatch setoff_pytorch_cifar.sh False False V1 post New_Pytorch_Cifar_Resnet_V1_post_slurm
#sbatch setoff_pytorch_cifar.sh False False V1 pre New_Pytorch_Cifar_Resnet_V1_pre_slurm
#sbatch setoff_pytorch_cifar.sh True False V1 post New_Pytorch_Cifar_Resnet_V1_USLC_post_slurm
#sbatch setoff_pytorch_cifar.sh True False V1 pre New_Pytorch_Cifar_Resnet_V1_USLC_pre_slurm
#sbatch setoff_og_pytorch_cifar.sh False False V1 pre New_Pytorch_Cifar_Resnet
#bash setoff_pytorch_cifar.sh False False V1 post Pytorch_Cifar_Resnet_V1_post | tee post_LCLS_cifar.out
#bash setoff_pytorch_cifar.sh False False V1 pre Pytorch_Cifar_Resnet_V1_pre | tee pre_LCLS_cifar.out
#bash setoff_grad.sh False False V1 post Record_Resnet_V1_frozen_spatial_post | tee record_grad_LCLS_post.out
#bash setoff_grad.sh False False V1 pre Record_Resnet_V1_frozen_spatial_pre | tee record_grad_LCLS_pre.out
