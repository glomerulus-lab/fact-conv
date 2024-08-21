sbatch setoff_general.sh attention_test.py  --k 0 --v 0 --q 0 --fixed 0 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 0 --q 0 --fixed 0 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 0 --v 1 --q 0 --fixed 0 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 1 --q 0 --fixed 0 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py  --k 0 --v 0 --q 0 --fixed 1 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 0 --q 0 --fixed 1 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 0 --v 1 --q 0 --fixed 1 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 1 --q 0 --fixed 1 --pre_processing mean --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py  --k 0 --v 0 --q 0 --fixed 0 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 0 --q 0 --fixed 0 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 0 --v 1 --q 0 --fixed 0 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 1 --q 0 --fixed 0 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py  --k 0 --v 0 --q 0 --fixed 1 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 0 --q 0 --fixed 1 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 0 --v 1 --q 0 --fixed 1 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py  --k 1 --v 1 --q 0 --fixed 1 --pre_processing sum --post_processing "" --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

