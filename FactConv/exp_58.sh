width=(4 2 1 0.5 0.25 0.125)


#bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18
#rank=1024 
rank=1
#bash setoff_general.sh long_cifar.py  --width 4 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}

#rank=512 
#bash setoff_general.sh long_cifar.py  --width 4 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 2 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}

#rank=256 
#sbatch setoff_general.sh long_cifar.py  --width 4 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 2 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 1 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}

rank=128 
#sbatch setoff_general.sh long_cifar.py  --width 4 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 2 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 1 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 0.5 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}


#rank=64 
#sbatch setoff_general.sh long_cifar.py  --width 4 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 2 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 1 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#sbatch setoff_general.sh long_cifar.py  --width 0.5 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}


#sbatch setoff_general.sh long_cifar.py --channel_k 16 --width 0.5 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_16
#sbatch setoff_general.sh long_cifar.py --channel_k 8 --width 0.5 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_8                  
bash setoff_general.sh long_cifar.py --channel_k 4 --width 0.5 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_4
#sbatch setoff_general.sh long_cifar.py --channel_k 1 --width 0.5 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_1
 
#
#long_cifar.py --channel_k 8 --width 4 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_8 
    
#sbatch setoff_general.sh long_cifar.py  --width 0.25 --channel_k $rank --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${rank}
#long_cifar.py --width 0.25 --channel_k 64 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_64                                                                     



#rank=(32 16 8 4 2 1)
#
#for i in  ${width[@]}
#do
#  
#  for j in  ${rank[@]}
#  do
#    sbatch setoff_general.sh long_cifar.py --channel_k $j --width $i --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18_lrdiag_${j}
#  done
#done
#

