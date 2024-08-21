width=(0 1)
niter=(1)
eps=(0.1 1e-4  0.0000001)
eps=(0.0)
eps=(1.0)
eps=(100000.0)

for i in ${width[@]}
do
  for j in ${niter[@]}
  do
    for k in ${eps[@]}
    do
      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 20 --niters 2 --eps $k --rank 200 --batchsize 200 --net factmlp --resample_freq 0 --epochs 100 
      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 20 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 0 --epochs 100
      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 20 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 3 --epochs 100
      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 2 --niters 2 --eps $k --rank 200 --batchsize 200 --net factmlp --resample_freq 0 --epochs 100
      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 2 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 0 --epochs 100
      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 2 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 3 --epochs 100
    done
  done
done

#width=(0 1)
#niter=(0)
#eps=(1e-4)
#
#for i in ${width[@]}
#do
#  for j in ${niter[@]}
#  do
#    for k in ${eps[@]}
#    do
#      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 20 --niters 2 --eps $k --rank 200 --batchsize 200 --net factmlp --resample_freq 0 --epochs 100 
#      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 20 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 0 --epochs 100
#      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 20 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 3 --epochs 100
#      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 2 --niters 2 --eps $k --rank 200 --batchsize 200 --net factmlp --resample_freq 0 --epochs 100
#      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 2 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 0 --epochs 100
#      sbatch setoff_general.sh svd_testing.py --double $i --test_svd $j --layer 2 --niters 2 --eps $k --rank 200 --batchsize 200 --net alignment --resample_freq 3 --epochs 100
#    done
#  done
#done
#
#
