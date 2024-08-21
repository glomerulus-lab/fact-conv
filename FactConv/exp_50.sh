epochs=(200)
seeds=(0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    #bash setoff_general.sh long_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 1 --net topk_width4_abs_pre_bn_aligned_resnet18 
    #bash setoff_general.sh long_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 0 --net topk_width4_abs_pre_bn_resnet18 
    bash setoff_general.sh nine_class_cifar.py  --width 4 --batchsize 1024 --num_epochs  $i --seed $j --resample 0 --net nineclass_width4_abs_pre_bn_resnet18 
    bash setoff_general.sh nine_class_cifar.py  --width 1 --batchsize 256 --num_epochs  $i --seed $j --resample 0 --net nineclass_width1_abs_pre_bn_resnet18 
    bash setoff_general.sh long_cifar.py  --width 1 --batchsize 256 --num_epochs  $i --seed $j --resample 0 --net topk_width1_abs_pre_bn_resnet18 
  done
done 
