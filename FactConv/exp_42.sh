epochs=(200)
#seeds=(0 1 2 3 4 5 6 7 8 9)
#seeds=(9 8 7 6 5)
seeds=(0)
for i in ${epochs[@]}
do
  for j in ${seeds[@]}
  do
    echo $i
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j  --net final_ok_retry_llnl_graphs_width4_abs_conv_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 2 --batchsize 512 --num_epochs  200 --seed $j  --net final_ok_retry_llnl_graphs_width2_abs_conv_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 1 --batchsize 256 --num_epochs  200 --seed $j --net final_ok_retry_llnl_graphs_width1_abs_conv_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --net final_ok_retry_llnl_graphs_width0.5_abs_conv_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --net final_ok_retry_llnl_graphs_width0.25_abs_conv_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --net final_ok_retry_llnl_graphs_width0.125_abs_conv_pre_bn_resnet18
    
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j  --net final_ok_retry_llnl_graphs_width4_abs_fact_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 2 --batchsize 512 --num_epochs  200 --seed $j  --net final_ok_retry_llnl_graphs_width2_abs_fact_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 1 --batchsize 256 --num_epochs  200 --seed $j  --net final_ok_retry_llnl_graphs_width1_abs_fact_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j  --net final_ok_retry_llnl_graphs_width0.5_abs_fact_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --net final_ok_retry_llnl_graphs_width0.25_abs_fact_pre_bn_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --net final_ok_retry_llnl_graphs_width0.125_abs_fact_pre_bn_resnet18

    #bash setoff_general.sh probe.py --name rqfree --optimize 1 --statistics 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net width1_abs_rqfree_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net width4_abs_rqfree_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree --optimize 1 --statistics 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net width2_abs_rqfree_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree --optimize 1 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net width0.5_abs_rqfree_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree --optimize 1 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net width0.25_abs_rqfree_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree --optimize 1 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net width0.125_abs_rqfree_pre_bn_resnet18 

    #bash setoff_general.sh probe.py --name eigh --optimize 1 --statistics 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net width1_abs_eigh_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net width4_abs_eigh_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh --optimize 1 --statistics 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net width2_abs_eigh_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh --optimize 1 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net width0.5_abs_eigh_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh --optimize 1 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net width0.25_abs_eigh_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh --optimize 1 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net width0.125_abs_eigh_pre_bn_resnet18 



    bash setoff_general.sh probe.py --name rqfree_lefthand --optimize 1 --statistics 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net width1_abs_rqfree_lefthand_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree_lefthand --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net width4_abs_rqfree_lefthand_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree_lefthand --optimize 1 --statistics 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net width2_abs_rqfree_lefthand_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree_lefthand --optimize 1 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net width0.5_abs_rqfree_lefthand_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree_lefthand --optimize 1 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net width0.25_abs_rqfree_lefthand_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name rqfree_lefthand --optimize 1 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net width0.125_abs_rqfree_lefthand_pre_bn_resnet18 


    #bash setoff_general.sh probe.py --name eigh_fixed --optimize 1 --statistics 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net width1_abs_eigh_fixed_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh_fixed --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net width4_abs_eigh_fixed_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh_fixed --optimize 1 --statistics 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net width2_abs_eigh_fixed_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh_fixed --optimize 1 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net width0.5_abs_eigh_fixed_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh_fixed --optimize 1 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net width0.25_abs_eigh_fixed_pre_bn_resnet18 
    #bash setoff_general.sh probe.py --name eigh_fixed --optimize 1 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net width0.125_abs_eigh_fixed_pre_bn_resnet18 



    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_eigh_pre_bn_aligned_resnet18 

    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_eigh_pre_bn_aligned_resnet18 

    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_eigh_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_eigh_pre_bn_aligned_resnet18 


    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_rqfree_lefthand_pre_bn_aligned_resnet18 

    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_rqfree_lefthand_pre_bn_aligned_resnet18 

    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_rqfree_lefthand_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_rqfree_lefthand_pre_bn_aligned_resnet18 



    
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net mydoing_resample_width4_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_rqfree_pre_bn_aligned_resnet18 

    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_rqfree_pre_bn_aligned_resnet18 

    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 1 --batchsize 256 --num_epochs  $i --seed $j --net resample_width1_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 4 --batchsize 1024 --num_epochs  $i --seed $j --net resample_width4_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 2 --batchsize 512 --num_epochs  $i --seed $j --net resample_width2_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.5 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.5_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.25 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.25_abs_rqfree_pre_bn_aligned_resnet18 
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --resample 1 --width 0.125 --batchsize 256 --num_epochs  $i --seed $j --net resample_width0.125_abs_rqfree_pre_bn_aligned_resnet18 


    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  retry_aligned_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18
    ##bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18
    ##bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18
    ##bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.25 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.25_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net  retry_aligned_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18
    ##bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18
    ##bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18
    ##bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 0.5 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.5_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --double 0 --optimize 1 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --double 0 --optimize 1 --statistics 0 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --double 0 --optimize 0 --statistics 1 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --double 0 --optimize 0 --statistics 0 --width 0.125 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width0.125_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 1 --batchsize 256 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width1_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 2 --batchsize 512 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width2_abs_pre_bn_aligned_resnet18

    #bash setoff_general.sh probe.py --optimize 1 --statistics 1 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 1 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 1 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
    #bash setoff_general.sh probe.py --optimize 0 --statistics 0 --width 4 --batchsize 1024 --num_epochs  200 --seed $j --resample 1 --net retry_aligned_llnl_graphs_width4_abs_pre_bn_aligned_resnet18
  done
done
