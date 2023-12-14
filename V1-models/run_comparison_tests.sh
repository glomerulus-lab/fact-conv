#!/bin/bash

module load python/3.8
source ../env/bin/activate


#python Test_Resnet_V1_CIFAR10.py --net fact | tee factnet.out
python Test_Resnet_V1_CIFAR10.py --net replace | tee replacenet.out
python Print_Resnet_Models.py | tee model_printout.out
