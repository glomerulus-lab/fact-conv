epochs=(200)
seeds=(0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    #bash setoff_general.sh attention_test.py  --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --post_processing mean_linear --k 1 --v 1 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --post_processing sum_linear --k 1 --v 1 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --pre_processing "sum" --post_processing linear --k 0 --v 0 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    bash setoff_general.sh attention_test.py --pre_processing sum --post_processing '' --k 0 --v 0 --q 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh attention_test.py --k 1 --v 1 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 0 --v 1 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 1 --v 0 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 0 --v 0 --q 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 1 --v 0 --q 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 0 --v 1 --q 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 1 --v 1 --q 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh attention_test.py --k 0 --v 0 --q 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh ensemble_test.py --name rqfree --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_rqfree_pre_bn_aligned_resnet18
  done
done
