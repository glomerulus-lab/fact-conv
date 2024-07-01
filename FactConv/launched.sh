#!/bin/bash                                          
#width=(0.125 0.25 0.5 1.0 2.0 4.0)
#width=(1)
#rank=(1.0 0.9 0.65 0.75 0.5 0.35 0.25 0.15 0.1 0.05)
rank=(1.0 0.75 0.5 0.25 0.1)
#seed=(1 2)
#width=(0.125)
seed=(0 1 2)

for i in ${rank[@]}
do
  for j in ${seed[@]}
  do
      #sbatch setoff.sh  --width $i --seed $j --net resnet18  
      #sbatch setoff.sh  --width $i --seed $j --net fact_resnet18
      #sbatch setoff.sh  --width $i --seed $j --net fact_us_resnet18
      #sbatch setoff.sh  --width $i --seed $j --net fact_uc_resnet18
      #sbatch setoff.sh  --width $i --seed $j --net fact_us_uc_resnet18
      # WHERE IS THIS VIVIAN 🧐🧐🤨🤨
      # NVM good job Vivian
      sbatch setoff.sh  $i $j resnet18_fact_lr-diag
      sbatch setoff.sh $i $j resnet18_fact_lowrank
      #sbatch setoff.sh  $i $j fact_diagchan_us_resnet18
      #sbatch setoff.sh  $i $j fact_diagchan_uc_resnet18
      #sbatch setoff.sh  $i $j fact_diagchan_us_uc_resnet18
  done
done

