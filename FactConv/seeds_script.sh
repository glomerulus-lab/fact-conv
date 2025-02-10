widths=(4 2 1)
ranks=(1 2 4 8 16 32 64 128 256 512 1024)
seeds=(1)
#width fact conv networks full rank error bars
#for i in ${ranks[@]}
#do
#  for j in ${widths[@]}
#  do
#i    sbatch setoff_general.sh old_alt_probe.py --width $j --channel_k $i --seed 1 --net final_pre_bn_resnet18_lrdiag --batchsize 1024 --resample 0
 #   sbatch setoff_general.sh old_alt_probe.py --width $j --channel_k $i --seed 2 --net final_pre_bn_resnet18_lrdiag --batchsize 1024 --resample 0
 # done
#done

#width rainbow networks full rank error bars
#seeds=(1 2 3 4)
#widths=(0.25 0.5 1 2)
#for i in ${widths[@]}
#do
#  for j in ${seeds[@]}
#  do
#    bash setoff_general.sh long_cifar.py --width $i --seed $j --net final_pre_bn_aligned_resnet18 --resample 1 --batchsize 1024
#  done
#done

#bash setoff_general.sh old_alt_probe.py --width 0.5 --seed 1 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1
#bash setoff_general.sh old_alt_probe.py --width 1 --seed 2 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1
#bash setoff_general.sh old_alt_probe.py --width 1 --seed 3 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1
#bash setoff_general.sh old_alt_probe.py --width 1 --seed 4 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1
#bash setoff_general.sh old_alt_probe.py --width 2 --seed 1 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1
#bash setoff_general.sh old_alt_probe.py --width 2 --seed 2 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1
#bash setoff_general.sh old_alt_probe.py --width 2 --seed 3 --net final_pre_bn_aligned_resnet18 --batchsize 1024 --resample 1

# PROBE AND ENSEMBLE TRAINSET 
seeds=(0 1 2 3 4)
widths=(1 2)
widths=(4)
for i in ${widths[@]}
do
  for j in ${seeds[@]}
  do
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_aligned_resnet9 --resample 1 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_aligned_resnet34 --resample 1 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_aligned_resnet50 --resample 1 --batchsize 1024 --width $i --seed $j
#
#    # probe and ensemble fact models when seeds are done
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 --width $i --seed $j
    sbatch setoff_general.sh alt_probe2.py --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 --width $i --seed $j
  done
done
### CURRENTLy$ RUNNING PROBE AND ENSEMBLE ON WIDTH 4 RESNET 50 separately
seeds=(0 1 2 3 4)
widths=(4)
for i in ${widths[@]}
do
  for j in ${seeds[@]}
  do
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_probe2.py --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_resnet9_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble2.py --net pre_bn_resnet34_fact --resample 0 --batchsize 1024 --width $i --seed $j
#    sbatch setoff_general.sh alt_probe.py --net pre_bn_resnet50_fact --resample 0 --batchsize 512 --width $i --seed $j
#    sbatch setoff_general.sh alt_ensemble.py --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 --width $i --seed $j
  done
done

# ALSO NEED TO SET OFF 2 MORE SEEDS OF RESNET50 WIDTH 4
#sbatch setoff_general.sh long_cifar.py --net pre_bn_aligned_resnet50 --resample 1 --batchsize 512 --width 4 --seed 3
#sbatch setoff_general.sh long_cifar.py --net pre_bn_aligned_resnet50 --resample 1 --batchsize 512 --width 4 --seed 4

#sbatch setoff_general.sh long_cifar.py --net pre_bn_resnet50_fact --resample 0 --batchsize 1024 --width 4 --seed 3
