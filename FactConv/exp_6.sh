#sbatch setoff_general.sh kmnist.py --net alignment --resample_freq 0 --epochs 100
#sbatch setoff_general.sh kmnist.py --net alignment --resample_freq 1 --epochs 100
#sbatch setoff_general.sh kmnist.py --net alignment --resample_freq 3 --epochs 100
#
#sbatch setoff_general.sh fmnist.py --net alignment --resample_freq 0 --epochs 100
#sbatch setoff_general.sh fmnist.py --net alignment --resample_freq 1 --epochs 100
#sbatch setoff_general.sh fmnist.py --net alignment --resample_freq 3 --epochs 100

sbatch setoff_general.sh mnist_firsthalf.py --net alignment  --first_half 0 --resample_freq 0
sbatch setoff_general.sh mnist_firsthalf.py --net alignment  --first_half 0 --resample_freq 1
sbatch setoff_general.sh mnist_firsthalf.py --net alignment  --first_half 0 --resample_freq 3


sbatch setoff_general.sh mnist_firsthalf.py --net alignment  --first_half 1 --resample_freq 0
sbatch setoff_general.sh mnist_firsthalf.py --net alignment  --first_half 1 --resample_freq 1
sbatch setoff_general.sh mnist_firsthalf.py --net alignment  --first_half 1 --resample_freq 3

