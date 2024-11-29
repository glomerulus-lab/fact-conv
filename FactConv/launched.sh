#!/bin/bash                                          
#width=(0.125 0.25 0.5 1.0 2.0 4.0)
#nonlins=("exp" "abs" "x^2" "1/x^2" "softplus" "identity" "relu")
width=(1)
rank=(1 2 4 8 16 32 64 128 256 512 1024)
seed=(0 1 2)
#seed=(3)

for i in ${width[@]}
do
  for j in ${rank[@]}
  do
    for k in ${seed[@]}
    do
      #sbatch setoff.sh  --width $i --seed $j --net resnet18  
      #sbatch setoff.sh  --width $i --seed $j --net fact_resnet18
      #sbatch setoff.sh  --width $i --seed $j --net fact_us_resnet18
      #sbatch setoff.sh  --width $i --seed $j --net fact_uc_resnet18
      #sbatch setoff.sh  --width $i --seed $j --net fact_us_uc_resnet18
      # WHERE IS THIS VIVIAN 🧐🧐🤨🤨
      # NVM good job Vivian
#      sbatch setoff.sh $i $j $k resnet18-fact-lr-K1-no-affines
#      sbatch setoff.sh $i $j $k resnet18-fact-lr-K1
#      sbatch setoff.sh $i $j $k resnet18-diagdom
#      sbatch setoff.sh $i $j $k resnet18-diagchan
#      sbatch setoff.sh $i $j $k resnet18-lr-diag
      sbatch setoff.sh $i $j $k wrn-lr-K1
      sbatch setoff.sh $i $j $k wrn-lr-diag
#      sbatch setoff.sh $i $j $k resnet18-fact-us
#      sbatch setoff.sh $i $j $k resnet18-fact-uc
#      sbatch setoff.sh $i $j $k resnet18-fact-usuc
      #sbatch setoff.sh  $i $j fact_diagchan_us_resnet18
      #sbatch setoff.sh  $i $j fact_diagchan_uc_resnet18
      #sbatch setoff.sh  $i $j fact_diagchan_us_uc_resnet18
    done
  done
done

