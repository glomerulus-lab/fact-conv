ranks=(1 2 4 8 16 32 64 128 256 512 1024)
seeds=(1)
for i in ${seeds[@]}
do
  for j in ${ranks[@]}
  do
    sbatch setoff_general.sh long_cifar.py --width 4 --net final_pre_bn_aligned_resnet18_lrdiag --channel_k $j  --seed $i --resample 1 --batchsize 1024
  done
done
