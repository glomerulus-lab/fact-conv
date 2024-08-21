epochs=(200)
seeds=(0) #1 2)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    #retry_r
   # bash setoff_general.sh long_cifar.py  --width 4 --batchsize 512 --num_epochs  $i --seed $j --resample 1 --net state_switch_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh double_cifar.py  --width 4 --batchsize 900 --num_epochs  $i --seed $j --resample 1 --net state_switch_2_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #7
    #sbatch setoff_general.sh double_cifar.py  --width 4 --batchsize 900 --num_epochs  $i --seed $j --resample 1 --net state_switch_3_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #sbatch setoff_general.sh double_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net state_switch_3_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #sbatch setoff_general.sh double_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net state_switch_4_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh double_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net state_switch_8_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    bash setoff_general.sh double_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net state_switch_9_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #
    #bash setoff_general.sh double_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net state_switch_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --bias 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --bias 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --bias 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --bias 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --bias 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j --resample 1 --net retry_llnl_graphs_width2_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_llnl_graphs_width1_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net retry_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j  --net final_retry_llnl_graphs_width4_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j  --net final_retry_llnl_graphs_width2_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j  --net final_retry_llnl_graphs_width1_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j  --net final_retry_llnl_graphs_width0.5_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j  --net final_retry_llnl_graphs_width0.25_abs_fact_pre_bn_resnet18 
    #
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width2_abs_pre_bn_resnet18 
    #0.25
    #bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width0.125_abs_pre_bn_resnet18 
  done
done
