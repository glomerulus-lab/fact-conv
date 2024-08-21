epochs=(200)
seeds=(0) # 1 2)
t=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --gmm 0 --bias 0  --width  4 --batchsize 1024 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    for k in ${t[@]}
    do
      #bash setoff_general.sh probe.py --t $k --optimize 1 --statistics 1 --bias 0 --gmm 0 --width  4 --batchsize 1024 --num_epochs 200 --seed $j --resample 1 --net state_switch_6_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
      
      bash setoff_general.sh probe.py --t $k --optimize 1 --statistics 1 --bias 0 --gmm 0 --width  4 --batchsize 1024 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    done
  
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --bias 0 --gmm 0 --width  4 --batchsize 1024 --num_epochs 200 --seed $j --resample 1 --net state_switch_7_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --bias 0 --gmm 0 --width  4 --batchsize 1024 --num_epochs 200 --seed $j --resample 1 --net state_switch_5_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --bias 1  --width  2 --batchsize 512 --num_epochs 200 --seed $j --resample 1 --net retry_bias_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18 
    #
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1  --width  2 --batchsize 512 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1  --width  1 --batchsize 256 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1  --width  0.5 --batchsize 256 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1  --width  0.25 --batchsize 256 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1  --width  0.125 --batchsize 256 --num_epochs 200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18 
    done
done
