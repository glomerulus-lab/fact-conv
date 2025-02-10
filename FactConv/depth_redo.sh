#sbatch setoff_general.sh long_cifar.py --width 1 --seed 3 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 2 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 4 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 1 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 2 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r
#
#
#sbatch setoff_general.sh long_cifar.py --width 1 --seed 1 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 1 --seed 2 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 1 --seed 3 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 1 --seed 4 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 1 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 2 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 3 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 4 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 1 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 2 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 3 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 4 --seed 4 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
#
#
#sbatch setoff_general.sh long_cifar.py --width 1 --seed 2 --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 -r
#sbatch setoff_general.sh long_cifar.py --width 2 --seed 2 --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 -r


sbatch setoff_general.sh long_cifar.py --width 0.25 --seed 3 --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 -r

sbatch setoff_general.sh long_cifar.py --width 0.25 --seed 2 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r
sbatch setoff_general.sh long_cifar.py --width 0.25 --seed 3 --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 -r

sbatch setoff_general.sh long_cifar.py --width 0.25 --seed 0 --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 -r
sbatch setoff_general.sh long_cifar.py --width 0.25 --seed 2 --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 -r
