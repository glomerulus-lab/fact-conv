epochs=(200)
seeds=(0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    bash setoff_general.sh alt_probe.py  --optimize 1 --statistics 1 --width 4 --batchsize 1000 --num_epochs  $i --seed $j --resample 1 --net topk_width4_abs_pre_bn_alt_aligned_resnet18 
  done
done 
