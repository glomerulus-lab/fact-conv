width=(32 64 128 256 512 1024 2048 2048*2)
width=(4096)


for i in ${width[@]}
do
  sbatch setoff_width_mnist.sh --width $i --net alignment --resample_freq 0 --epochs 100
  sbatch setoff_width_mnist.sh --width $i --net alignment --resample_freq 1 --epochs 100
  sbatch setoff_width_mnist.sh --width $i --net alignment --resample_freq 2 --epochs 100
  sbatch setoff_width_mnist.sh --width $i --net alignment --resample_freq 3 --epochs 100
done
