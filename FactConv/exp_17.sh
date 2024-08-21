epochs=(64 128 256 512)
rank=(64 128 256 500)

for j in ${rank[@]} 
do
  for i in ${epochs[@]}
  do
    echo $i
    #sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --net fact_pre_bn_resnet18_resample
    #sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --resample 1 --net fact_pre_bn_resnet18_resample
    #sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --net pre_bn_resnet18
    #sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --net fact_pre_bn_resnet18

    sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --net fact_pre_bn_resnet9_resample
    sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --resample 1 --net fact_pre_bn_resnet9_resample
    sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --net pre_bn_resnet9
    sbatch setoff_general.sh long_cifar.py --rank $j  --batchsize  $i --seed 0 --net fact_pre_bn_resnet9
  done
done
