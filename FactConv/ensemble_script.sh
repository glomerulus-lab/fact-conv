ranks=(1 2 4 8 16 32 64 128 256 512 1024 2048)
widths=(0.25 0.5 1 2 4)
seeds=(0 1 2 3 4)
for i in ${widths[@]}
do
  for j in ${seeds[@]}
  do
  sbatch setoff_general.sh alt_ensemble.py --width $i --seed $j --batchsize 1024 --resample 1 --net final_pre_bn_aligned_resnet18
  done
done

