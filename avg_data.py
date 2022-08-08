#Python script to take the average of 3 trials of BN_V1_V1_Linear run
import numpy as np
import argparse
import os
import torch
from scipy.stats import sem

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--name', type=str, default='new model', help='filename for saved model')
    parser.add_argument('--model', type=str, default="BN_V1_V1_Linear", help='name of model')
    parser.add_argument('--trials', type=int, default=3, help='number of trials there is data for')
    args = parser.parse_args()
    
    src = "/research/harris/vivian/v1-models/saved-models/" + args.model + "/" + args.name
    loss = []
    accuracy = []
    for i in range(args.trials):
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
    print("Num trials: " + str(args.trials))
    print("Avg loss: " + str(avg_loss) + " +/- " + str(loss_err))
    print("Avg accuracy: " + str(avg_acc) + " +/- " + str(acc_err))
    
    
    
    