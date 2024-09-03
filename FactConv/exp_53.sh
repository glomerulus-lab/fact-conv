epochs=(200)
seeds=(0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i 
    #bash setoff_general.sh long_cifar.py  --width 4 --batchsize 250 --num_epochs  $i --seed $j --resample 1 --net new_topk_width4_abs_pre_bn_alt_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py  --width 4 --batchsize 250 --num_epochs  $i --seed $j --resample 1 --net retest_new_topk_width4_abs_pre_bn_aligned_resnet18 
    #
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net dropout_2_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh alt_probe.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net dropout_2_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net dropout_4_pre_bn_alt_aligned_resnet18
    
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net dropout_5_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net alignment_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net alignment_2_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net alignment_3_pre_bn_alt_aligned_resnet18
    #
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net alignment_6_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh alt_probe.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net alignment_pre_bn_alt_aligned_resnet18
    #
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net lignment_pre_bn_alt_aligned_resnet18
    
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed  0 --resample 1 --net dropout_7_pre_bn_alt_aligned_resnet18
    bash setoff_general.sh long_cifar.py --quench 0 --global 0.4 --gen_dropout .7 --ref_dropout .3 --momentum 1 --width 4 --batchsize 1000 --seed 0 --resample 1 --net dropout_8_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net dropout_3_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh alt_probe.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net dropout_3_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh alt_probe.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net dropout_pre_bn_alt_aligned_resnet18
    #bash setoff_general.sh alt_probe.py --width 4 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18
    #sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 1000 --seed 0 --resample 1 --net pre_bn_aligned_resnet18
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_aligned_resnet18
  done
done 
