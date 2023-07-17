#Python script to take the average of 3 trials of each model run and put in DataFrame
import numpy as np
import argparse
import os
import torch
from scipy.stats import sem

if __name__ == '__main__':
    names = ["BN_V1_V1_Linear_bias_true", "Uniform_Control_bias_true", "Gaussian_Control_bias_true", "Scattering_Linear_Control"]
    datasets = ["CIFAR10", "CIFAR100", "MNIST", "CIFAR10_50_Samples"]
    
    file = open("figure_data.txt", 'w')
    header = "model dataset loss loss_err accuracy accuracy_err"
    file.write(header)
    file.write("\n")  
    
    for model in names:
        for data in datasets:
            
            src = "/research/harris/vivian/v1-models/saved-models/" + str(data) + "/" + str(model)
            loss = []
            accuracy = []
            for i in range(3):
                n=i+1
                os.chdir(src + "/trial_" + str(n))

                loss_trial = torch.load("loss.pt")[-1]
                accuracy_trial = torch.load("accuracy.pt")[-1]

                loss.append(loss_trial)
                accuracy.append(accuracy_trial)

    
            avg_loss = np.round_(np.mean(loss), decimals=4)
            avg_acc = np.round_(np.mean(accuracy), decimals=2)
            loss_err = np.round_(sem(loss), decimals=4)
            acc_err = np.round_(sem(accuracy), decimals=2)
            

            string = str(model) + " " + str(data) + " " + str(avg_loss) + " " + str(loss_err) + " " + str(avg_acc) + " " + str(acc_err)
            file.write(string)
            file.write("\n")
            