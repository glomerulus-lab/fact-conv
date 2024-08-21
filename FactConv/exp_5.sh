
#width=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#width=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
width=(0 1 5 10 20)

for i in ${width[@]}
do
  sbatch setoff_general.sh mnist_layers.py --layer $i --net factmlp  --resample_freq 0 --epochs 400
  sbatch setoff_general.sh mnist_layers.py --layer $i --net alignment   --resample_freq 0 --epochs 400
  sbatch setoff_general.sh mnist_layers.py --layer $i --net alignment --resample_freq 1 --epochs 400
  sbatch setoff_general.sh mnist_layers.py --layer $i --net alignment --resample_freq 3 --epochs 400
done
