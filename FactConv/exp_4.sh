sbatch setoff_general.sh kmnist.py --net alignment --resample_freq 0 --epochs 100
sbatch setoff_general.sh kmnist.py --net alignment --resample_freq 1 --epochs 100
sbatch setoff_general.sh kmnist.py --net alignment --resample_freq 3 --epochs 100

sbatch setoff_general.sh fmnist.py --net alignment --resample_freq 0 --epochs 100
sbatch setoff_general.sh fmnist.py --net alignment --resample_freq 1 --epochs 100
sbatch setoff_general.sh fmnist.py --net alignment --resample_freq 3 --epochs 100
