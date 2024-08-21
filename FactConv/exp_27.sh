epochs=(200)
seeds=(0 1 2)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18

    bash setoff_general.sh probe.py --width 0.125 --batchsize 256 --num_epochs 200 --seed $j --net llnl_graphs_width0.125_abs_fact_pre_bn_resnet18
    bash setoff_general.sh probe.py --width 0.25 --batchsize 256 --num_epochs 200 --seed $j --net llnl_graphs_width0.25_abs_fact_pre_bn_resnet18
    bash setoff_general.sh probe.py --width 0.5 --batchsize 256 --num_epochs 200 --seed $j --net llnl_graphs_width0.5_abs_fact_pre_bn_resnet18
    bash setoff_general.sh probe.py --width 1 --batchsize 256 --num_epochs 200 --seed $j --net llnl_graphs_width1_abs_fact_pre_bn_resnet18
    bash setoff_general.sh probe.py --width 2 --batchsize 512 --num_epochs 200 --seed $j --net llnl_graphs_width2_abs_fact_pre_bn_resnet18
    bash setoff_general.sh probe.py --width 4 --batchsize 1024 --num_epochs 200 --seed $j --net llnl_graphs_width4_abs_fact_pre_bn_resnet18
    
    #bash setoff_general.sh width_small_probe.py --double 1 --optimize 1 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh width_small_probe.py --double 1 --optimize 1 --statistics 0 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh width_small_probe.py --double 1 --optimize 0 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh width_small_probe.py --double 1 --optimize 0 --statistics 0 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width1_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width1_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width1_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width1_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width2_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width2_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width2_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width2_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net  llnl_graphs_width4_abs_pre_bn_aligned_resnet18
  done
done
