epochs=(200)
seeds=(0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    bash setoff_general.sh attention_test.py --num_heads 8 --k 1 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
  done
done 
