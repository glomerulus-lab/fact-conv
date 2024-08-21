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

    sbatch setoff_general.sh double_cifar.py --weight1 0.5 --weight2 0.5 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.5_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.4 --weight2 0.6 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.4_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.3 --weight2 0.7 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.3_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.2 --weight2 0.8 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.2_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.1 --weight2 0.9 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.1_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.0 --weight2 1.0 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.0_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.6 --weight2 0.4 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.6_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.7 --weight2 0.3 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.7_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.8 --weight2 0.2 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.8_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 0.9 --weight2 0.1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_0.9_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh double_cifar.py --weight1 1.0 --weight2 0.0 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net width4_1.0_abs_pre_bn_aligned_resnet18 
 
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
