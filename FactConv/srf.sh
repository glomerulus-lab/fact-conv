ranks=(1 2 4 8 16 32 64 128 256 512 1024)
#ranks=(1)
for i in ${ranks[@]}
do
  sbatch setoff_long.sh long_cifar.py --width 4 --channel_k $i --seed 2 --net final_pre_bn_resnet18_lrdiag --batchsize 1024 --resample 0
done

ranks=(1 2 4 8 16 32 64 128 256 512)
#ranks=(1 2 4 8 16 32)
#ranks=(64 128 256 512)
for i in ${ranks[@]}
do
  sbatch setoff_long.sh long_cifar.py --width 2 --channel_k $i --seed 2 --net final_pre_bn_resnet18_lrdiag --batchsize 1024 --resample 0
done

ranks=(1 2 4 8 16 32 64 128 256)
for i in ${ranks[@]}
do
  sbatch setoff_long.sh long_cifar.py --width 1 --channel_k $i --seed 2 --net final_pre_bn_resnet18_lrdiag --batchsize 1024  --resample 0
done
