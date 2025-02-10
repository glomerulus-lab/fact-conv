# rainbow models
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 -r
## srf baselines
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 -r
## trad baselines
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_resnet9 --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_resnet34 --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_resnet9 --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_resnet34 --resample 0 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 0 --net pre_bn_resnet50 --resample 0 --batchsize 1024 -r

# 5 seeds total for width 1 each depth
seeds=(0 1 2 3 4)
widths=(1 2)
for i in ${seeds[@]}
do
#  for j in ${widths[@]}
#  do
#    sbatch setoff_general.sh long_cifar.py --width $j --seed $i --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 -r
#    sbatch setoff_general.sh long_cifar.py --width $j --seed $i --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 -r
#    sbatch setoff_general.sh long_cifar.py --width $j --seed $i --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 -r
#  done
#  sbatch setoff_general.sh long_cifar.py --width 4 --seed $i --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 -r
#  sbatch setoff_general.sh long_cifar.py --width 4 --seed $i --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 -r
  sbatch setoff_general.sh long_cifar.py --width 4 --seed $i --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 -r
done

#sbatch setoff_general.sh long_cifar.py --width 4 --seed 0 --net pre_bn_resnet50_fact --resample 0 --batchsize 512 -r

# 5 seeds total for each width at each depth 
#widths=(0.25 0.5)
#seeds=(0 1 2 3 4)
#for i in ${widths[@]}
#do
#  for j in ${seeds[@]}
#  do
#    sbatch setoff_general.sh long_cifar.py --width $i --seed $j --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#    sbatch setoff_general.sh long_cifar.py --width $i --seed $j --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#    sbatch setoff_general.sh long_cifar.py --width $i --seed $j --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 -r
#  done
#done
#
