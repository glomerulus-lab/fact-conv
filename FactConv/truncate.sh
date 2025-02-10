seeds=(0 1 2)
ranks=(500 1000 2048)
#ranks=(512 1024 2048)

for i in ${ranks[@]}
do
  for j in ${seeds[@]}
  do
    sbatch setoff_general.sh eval_three_mua.py --width 2048 --net alignmlp --seed $j --nonlin relu --align_rank $i
    # sbatch setoff_general.sh eval_three_mua.py --width 2048 --net alignmlp --seed $j --nonlin sigmoid --align_rank $i
  done
done
