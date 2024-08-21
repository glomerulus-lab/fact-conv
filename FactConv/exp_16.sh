epochs=(200 400 600 800)
for i in ${epochs[@]}
do
  echo $i
  sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net fact_pre_bn_resnet18_resample
  sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --resample 1 --net fact_pre_bn_resnet18_resample
  sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net pre_bn_resnet18
  sbatch setoff_general.sh long_cifar.py --num_epochs  $i --seed 0 --net fact_pre_bn_resnet18
done
