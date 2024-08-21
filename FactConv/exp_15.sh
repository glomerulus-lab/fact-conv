
for i in {0..2} 
do
  echo $i
  sbatch setoff_general.sh cifar.py --seed $i --net fact_pre_bn_resnet18_resample
  sbatch setoff_general.sh cifar.py --seed $i --double 1 --net fact_pre_bn_resnet18_resample
  sbatch setoff_general.sh cifar.py --seed $i --resample 1 --net fact_pre_bn_resnet18_resample
  sbatch setoff_general.sh cifar.py --seed $i --resample 1 --double 1 --net fact_pre_bn_resnet18_resample
  #sbatch setoff_general.sh cifar.py --seed $i --net pre_bn_resnet18_resample
  #sbatch setoff_general.sh cifar.py --seed $i --double 1 --net pre_bn_resnet18_resample
  sbatch setoff_general.sh cifar.py --seed $i --net pre_bn_resnet18
  sbatch setoff_general.sh cifar.py --seed $i --net fact_pre_bn_resnet18
  sbatch setoff_general.sh cifar.py --seed $i --net resnet18
  sbatch setoff_general.sh cifar.py --seed $i --net fact_resnet18
done
