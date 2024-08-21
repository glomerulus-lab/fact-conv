epochs=(200)
seeds=(0 1 2)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
  done
done
