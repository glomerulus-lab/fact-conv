
#width=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#width=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#width=(0 1 5 10 20)
#width=(1000 500 200  )
width=(200) # 500 200  )
#width=(200 500 700 1000) # 500 200  )
#niter=(2 5 10  20 50 100)
niter=(2)
for i in ${width[@]}
do
  for j in ${niter[@]}
  do
    sbatch setoff_general.sh final_kmnist.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net factmlp --resample_freq 0 --epochs 1000 --resume 1
    sbatch setoff_general.sh final_kmnist.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net alignment --resample_freq 0 --epochs 1000 --resume 1
    sbatch setoff_general.sh final_kmnist.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net alignment --resample_freq 3 --epochs 1000 --resume 1
    sbatch setoff_general.sh final_fashmnist.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net factmlp --resample_freq 0 --epochs 1000 --resume 1
    sbatch setoff_general.sh final_fashmnist.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net alignment --resample_freq 0 --epochs 1000 --resume 1
    sbatch setoff_general.sh final_fashmnist.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net alignment --resample_freq 3 --epochs 1000 --resume 1
    sbatch setoff_general.sh test_final.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net factmlp --resample_freq 0 --epochs 1000 --resume 1
    sbatch setoff_general.sh test_final.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net alignment --resample_freq 0 --epochs 1000 --resume 1
    sbatch setoff_general.sh test_final.py --layer 20 --niters $j --eps 0.000 --rank 200 --batchsize $i --net alignment --resample_freq 3 --epochs 1000 --resume 1

  done
done
