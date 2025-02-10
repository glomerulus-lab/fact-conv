seeds=(0 1 2)
ranks=(500 1000)
for i in ${ranks[@]}
do
  for j in ${seeds[@]}
  do
    sbatch setoff_general.sh mnist_ensemble.py --width 2048 --net alignmlp --seed $j --nonlin relu --align_rank $i
    sbatch setoff_general.sh mnist_ensemble.py --width 2048 --net alignmlp --seed $j --nonlin sigmoid --align_rank $i
    sbatch setoff_general.sh mnist_probe.py --width 2048 --net alignmlp --seed $j --nonlin relu --align_rank $i
    sbatch setoff_general.sh mnist_probe.py --width 2048 --net alignmlp --seed $j --nonlin sigmoid --align_rank $i
  done
done
