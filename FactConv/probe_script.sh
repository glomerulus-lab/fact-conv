# rainbow unadapted trainset
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --resample 1 --net pre_bn_aligned_resnet9
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --resample 1 --net pre_bn_aligned_resnet34
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --resample 1 --net pre_bn_aligned_resnet50
#sbatch setoff_general.sh alt_probe.py --width 4.0 --batchsize 1024 --resample 1 --net pre_bn_aligned_resnet9
#sbatch setoff_general.sh alt_probe.py --width 4.0 --batchsize 1024 --resample 1 --net pre_bn_aligned_resnet34


# trad
#sbatch setoff_general.sh alt_probe.py --width 1.0 --batchsize 1024 --net pre_bn_resnet9
#sbatch setoff_general.sh alt_probe.py --width 1.0 --batchsize 1024 --net pre_bn_resnet34
#sbatch setoff_general.sh alt_probe.py --width 1.0 --batchsize 1024 --net pre_bn_resnet50
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --net pre_bn_resnet9
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --net pre_bn_resnet34
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --net pre_bn_resnet50
#sbatch setoff_general.sh alt_probe.py --width 4.0 --batchsize 1024 --net pre_bn_resnet9
#sbatch setoff_general.sh alt_probe.py --width 4.0 --batchsize 1024 --net pre_bn_resnet34
   

# fact
#sbatch setoff_general.sh alt_probe.py --width 1.0 --batchsize 1024 --net pre_bn_resnet9_fact
#sbatch setoff_general.sh alt_probe.py --width 1.0 --batchsize 1024 --net pre_bn_resnet34_fact
#sbatch setoff_general.sh alt_probe.py --width 1.0 --batchsize 1024 --net pre_bn_resnet50_fact
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --net pre_bn_resnet9_fact
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --net pre_bn_resnet34_fact
#sbatch setoff_general.sh alt_probe.py --width 2.0 --batchsize 1024 --net pre_bn_resnet50_fact
#sbatch setoff_general.sh alt_probe.py --width 4.0 --batchsize 1024 --net pre_bn_resnet9_fact
#sbatch setoff_general.sh alt_probe.py --width 4.0 --batchsize 1024 --net pre_bn_resnet34_fact

#bash setoff_general.sh alt_probe.py --seed 0 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
#bash setoff_general.sh alt_probe.py --seed 1 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
#bash setoff_general.sh alt_probe.py --seed 2 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
#bash setoff_general.sh alt_probe.py --seed 3 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
#bash setoff_general.sh alt_probe.py --seed 4 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
bash setoff_general.sh alt_ensemble.py --seed 3 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
bash setoff_general.sh alt_ensemble.py --seed 4 --width 4.0 --batchsize 512 --resample 1 --net pre_bn_aligned_resnet50
