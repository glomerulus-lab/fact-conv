epochs=(200)
seeds=(0 1 2)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $j
    #bash setoff_general.sh long_cifar.py --width 4 --batchsize 1024 --num_epochs 200 --seed $j --net llnl_graphs_width4_abs_fact_pre_bn_resnet18
    bash setoff_general.sh long_cifar.py --width 0.125 --batchsize 256 --num_epochs 200 --seed $j --net llnl_graphs_width0.125_abs_fact_pre_bn_resnet18
  done
done
