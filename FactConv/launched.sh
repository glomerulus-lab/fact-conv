#!/bin/bash                                          
width=(0.125 0.25 0.5 1.0 2.0 4.0)
seed=(0 1 2)


for i in ${width[@]}
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
      sbatch setoff.sh  --width $i --seed $j --net fact_diag_resnet18
      sbatch setoff.sh  --width $i --seed $j --net fact_diag_us_resnet18
      sbatch setoff.sh  --width $i --seed $j --net fact_diag_uc_resnet18
      sbatch setoff.sh  --width $i --seed $j --net fact_diag_us_uc_resnet18
    done
done

