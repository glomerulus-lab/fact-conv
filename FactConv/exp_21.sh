epochs=(200)
seeds=(0 1 2)
#bash setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --resample 1 --num_epochs  200  --seed 0 --net width0.25_abs_pre_bn_aligned_resnet18
#bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --resample 1 --num_epochs  200  --seed 0 --net width0.125_abs_pre_bn_aligned_resnet18
#
#bash setoff_general.sh long_cifar.py --width 0.5 --batchsize 128 --resample 1 --num_epochs  200  --seed 0 --net width0.5_abs_pre_bn_aligned_resnet18

#bash setoff_general.sh long_cifar.py --width 0.25 --batchsize 64 --resample 1 --num_epochs  200  --seed 0 --net width0.25_abs_pre_bn_aligned_resnet18
#
#bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 32 --resample 1 --num_epochs  200  --seed 0 --net width0.125_abs_pre_bn_aligned_resnet18
#bash setoff_general.sh long_cifar.py --width 0.25 --batchsize 128 --resample 1 --num_epochs  200  --seed 0 --net width0.25_abs_pre_bn_aligned_resnet18
bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 128 --resample 1 --num_epochs  200  --seed 0 --net width0.125_abs_pre_bn_aligned_resnet18
bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 64 --resample 1 --num_epochs  200  --seed 0 --net width0.125_abs_pre_bn_aligned_resnet18
#do
#  for j in ${seeds[@]}
#  do
#    echo $i
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0  --net pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net fact_pre_bn_resnet18_resample
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net fact_pre_bn_resnet18_resample
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net pre_bn_resnet18
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net fact_pre_bn_resnet18
#  #sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed 0 --net width2_abs_fact_pre_bn_resnet18
#  #
#  #sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 256 --num_epochs  $i --seed 0 --net width2_abs_fact_pre_bn_resnet18
#  #sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 256 --num_epochs  $i --seed 0 --net width2_abs_pre_bn_aligned_resnet18
#  #sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 256 --num_epochs  $i --seed 0 --resample 1 --net width2_abs_final_pre_bn_aligned_resnet18 
#
#
#    sbatch setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net width4_abs_fact_pre_bn_resnet18
#    sbatch setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net width4_abs_pre_bn_aligned_resnet18
#    sbatch setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_abs_pre_bn_aligned_resnet18 
#
#    #sbatch setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net width0.125_abs_fact_pre_bn_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net width0.125_abs_pre_bn_aligned_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net width0.125_abs_pre_bn_aligned_resnet18 
#
#    #sbatch setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net width0.25_abs_fact_pre_bn_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net width0.25_abs_pre_bn_aligned_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net width0.25_abs_pre_bn_aligned_resnet18 
#
#    #sbatch setoff_general.sh long_cifar.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net width0.5_abs_fact_pre_bn_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net width0.5_abs_pre_bn_aligned_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net width0.5_abs_pre_bn_aligned_resnet18 
##
#    #sbatch setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --net width1_abs_fact_pre_bn_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --net width1_abs_pre_bn_aligned_resnet18
#    #sbatch setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net width1_abs_pre_bn_aligned_resnet18 
#
#    sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j --net width2_abs_fact_pre_bn_resnet18
#    sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j --net width2_abs_pre_bn_aligned_resnet18
#    sbatch setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j --resample 1 --net width2_abs_pre_bn_aligned_resnet18 
#
#
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --batchsize 512 --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net abs_final_pre_bn_aligned_resnet18 
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0  --net abs_final_pre_bn_aligned_resnet18 
#  #post
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net affine_false_bn_aligned_resnet9 
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0  --net affine_false_bn_aligned_resnet9 
#
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net non_linear_added_pre_bn_aligned_resnet9 
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0  --net non_linear_added_pre_bn_aligned_resnet9 
#
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net bn_resnet9_resample
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net fact_pre_bn_resnet9_resample
#
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net pre_bn_resnet9
#  #sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net fact_pre_bn_resnet9
#  done
#done
