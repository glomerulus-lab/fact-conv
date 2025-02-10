seeds=(0 1 2)
ranks=(1 2 4 8 16 32 64 128 256 512 1024)
#for i in ${ranks[@]}
#do
#  for j in ${seeds[@]}
#  do
#    sbatch setoff_general.sh alt_ensemble2.py --width 4.0 --net final_pre_bn_aligned_resnet18_lrdiag --channel_k $i --seed $j --resample 1 --batchsize 1024
#  done
#done
for j in ${seeds[@]}
do
  sbatch setoff_general.sh alt_ensemble2.py --width 4.0 --net final_pre_bn_aligned_resnet18 --seed $j --resample 1 --batchsize 1024 --channel_k 2048
done

