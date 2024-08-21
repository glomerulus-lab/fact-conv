epochs=(200)
seeds=(0 1 2)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    #retry_resample 
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net retry_llnl_graphs_width4_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width4_abs_pre_bn_resnet18 
    #bash setoff_general.sh long_cifar.py --width 2 --batchsize 512 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width2_abs_pre_bn_resnet18 
    #0.25
    bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs  $i --seed $j  --net retry_llnl_graphs_width0.125_abs_pre_bn_resnet18 
  done
done
