#!/bin/bash

sbatch setoff.sh resnet18 True fact_resnet_width_1 0.1 1 1024
sbatch setoff.sh resnet18 False conv_resnet_width_1 0.1 1 1024
sbatch setoff.sh alexnet True fact_alexnet_width_1 0.01 1 1024
sbatch setoff.sh alexnet False conv_alexnet_width_1 0.01 1 1024

sbatch setoff_width2.sh resnet18 True fact_resnet_width_2 0.1 2 512
sbatch setoff_width4.sh resnet18 True fact_resnet_width_4 0.1 4 256
