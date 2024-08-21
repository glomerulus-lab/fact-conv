epochs=(200)
seeds=(0) #1 2)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    sbatch setoff_general.sh svhn.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net svhn_width4_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh svhn.py --width 2 --batchsize 512 --num_epochs  $i --seed $j --resample 1 --net svhn_width2_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh svhn.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net svhn_width1_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net svhn_width0.5_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net svhn_width0.25_abs_pre_bn_aligned_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --resample 1 --net svhn_width0.125_abs_pre_bn_aligned_resnet18 

    #
    sbatch setoff_general.sh svhn.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j  --net svhn_width4_abs_conv_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 2 --batchsize 512 --num_epochs  $i --seed $j  --net svhn_width2_abs_conv_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --net svhn_width1_abs_conv_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net svhn_width0.5_abs_conv_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j  --net svhn_width0.25_abs_conv_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j  --net svhn_width0.125_abs_conv_pre_bn_resnet18 

    sbatch setoff_general.sh svhn.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j  --net svhn_width4_abs_fact_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 2 --batchsize 512 --num_epochs  $i --seed $j  --net svhn_width2_abs_fact_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --net svhn_width1_abs_fact_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net svhn_width0.5_abs_fact_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j  --net svhn_width0.25_abs_fact_pre_bn_resnet18 
    sbatch setoff_general.sh svhn.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j  --net svhn_width0.125_abs_fact_pre_bn_resnet18 
 



    #
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width4_abs_fact_pre_bn_resnet18 
    #
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width4_abs_conv_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width2_abs_conv_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width1_abs_conv_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width0.5_abs_conv_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width0.25_abs_conv_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width0.125_abs_conv_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 1 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width1_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width0.125_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net final_ok_retry_llnl_graphs_width0.125_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 1 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net two_times_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --bias 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
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
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j  --net final_retry_llnl_graphs_width4_abs_fact_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width2_abs_pre_bn_resnet18 
    #0.25
    #bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width0.125_abs_pre_bn_resnet18 
  done
done
