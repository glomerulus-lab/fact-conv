#python program to print command to run each model 3 times
import numpy as np

models = ["BN_V1_V1_Linear", "Uniform", "Gaussian"]
datasets = ["CIFAR10", "CIFAR100", "CIFAR10_Small_Sample", "MNIST"]
num_combinations = len(models) * len(datasets)

i=0
for m in models:
    for d in datasets:
        i += 1
        if i <= num_combinations / 2:
            for t in range(3):
                print("python3 {}_{}.py --device=0 --trial={} --bias=True --name='{}_bias_true'".format(m, d, t+1, m))
        else:
            for t in range(3):
                print("python3 {}_{}.py --device=1 --trial={} --bias=True --name='{}_bias_true'".format(m, d, t+1, m))