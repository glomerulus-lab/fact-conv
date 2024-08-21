sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 0 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 0 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 0 --v 1 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 2 --k 1 --v 1 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18



sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 0 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 0 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 0 --v 1 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --num_heads 4 --k 1 --v 1 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18




sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing sum_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 0 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18

sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 1 --pre_processing "" --post_processing mean_linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 0 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 1 --pre_processing mean --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 0 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 0 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 0 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 0 --v 1 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
sbatch setoff_general.sh attention_test.py --nums_heads 8 --k 1 --v 1 --q 1 --fixed 1 --pre_processing sum --post_processing linear --resample 1 --width 4 --batchsize 1024 --seed 0 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18


