
#width=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#width=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#width=(0 1 5 10 20)
width=(1000 500 200  )
width=(500) # 500 200  )
width=(700 1000) # 500 200  )
niter=(2 5 10  20 50 100)
niter=(2)

for i in ${width[@]}
do
  for j in ${niter[@]}
  do
  #sbatch setoff_general.sh test_deep.py --layer $i --rank 200 --batchsize 200 --net factmlp --resample_freq 0 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer $i --rank 200 --batchsize 200 --net alignment --resample_freq 0 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer $i --rank 200 --batchsize 200 --net alignment --resample_freq 1 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer $i --rank 200 --batchsize 200 --net alignment --resample_freq 3 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer 20 --niters 20 --eps 0.00001 --rank 200 --batchsize $i --net alignment --resample_freq 0 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer 20 --niters 20 --eps 0.00001 --rank 200 --batchsize $i --net alignment --resample_freq 1 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer 20 --niters 20 --eps 0.00001 --rank 200 --batchsize $i --net alignment --resample_freq 3 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer 20 --niters $j --eps 0.0001 --rank 200 --batchsize $i --net alignment --resample_freq 1 --epochs 100
  #sbatch setoff_general.sh test_deep.py --layer 20 --niters $j --eps 0.0001 --rank 200 --batchsize $i --net alignment --resample_freq 1 --epochs 100
  # 
    #sbatch setoff_general.sh test_deep.py --layer 20 --niters $j --eps 0.0001 --rank 200 --batchsize $i --net alignment --resample_freq 0 --epochs 100
    #sbatch setoff_general.sh test_deep.py --layer 20 --niters $j --eps 0.0001 --rank 200 --batchsize $i --net alignment --resample_freq 3 --epochs 100
    if (( $i >  400 )); then
      sbatch setoff_general.sh test_deep.py --layer 20 --niters $j --eps 0.0001 --rank 500 --batchsize $i --net alignment --resample_freq 0 --epochs 100
      sbatch setoff_general.sh test_deep.py --layer 20 --niters $j --eps 0.0001 --rank 500 --batchsize $i --net alignment --resample_freq 3 --epochs 100
    fi
  done
done
