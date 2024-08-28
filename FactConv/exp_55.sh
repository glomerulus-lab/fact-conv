width=(4 2 1 0.5 0.25 0.125)
seed=(0 1 2)
for i in  ${width[@]}
do
  sbatch setoff_general.sh long_cifar.py --width $i --batchsize 1000 --seed 0 --resample 1 --net final_pre_bn_alt_aligned_resnet18
  for j in  ${seed[@]}
  do
    sbatch setoff_general.sh long_cifar.py --width $i --batchsize 1000 --seed $j --resample 0 --net final_pre_bn_resnet18
    sbatch setoff_general.sh long_cifar.py --width $i --batchsize 1000 --seed $j --resample 0 --net final_pre_bn_resnet18_fact
  done
done
