
width=(1 2 5 10 20 50 100 200 500 1000 2000)
rank=(10 50 100 200 400)

for j in ${rank[@]}
do
  for i in ${width[@]}
  do
    sbatch setoff_width_mnist.sh --batchsize $i --rank $j --net alignment --resample_freq 0 --epochs 100
    sbatch setoff_width_mnist.sh --batchsize $i --rank $j --net alignment --resample_freq 1 --epochs 100
   # sbatch setoff_width_mnist.sh --batchsize $i --rank $j --net alignment --resample_freq 2 --epochs 100
    sbatch setoff_width_mnist.sh --batchsize $i --rank $j --net alignment --resample_freq 3 --epochs 100
  done
done
