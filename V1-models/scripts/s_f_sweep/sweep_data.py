#Python script to take the average of 3 trials of BN_V1_V1_Linear run
import numpy as np
import argparse
import os
import torch
from scipy.stats import sem

if __name__ == '__main__':
    f_arr = np.array([0.1, 0.5, 1, 2, 3, 5, 7])
    s_arr = np.array([1, 2, 3, 4, 5, 6, 10])
    
    file = open("sweep_data.txt", 'w')
    header = "s f loss loss_err accuracy accuracy_err"
    file.write(header)
    file.write("\n")  
    
    for f in f_arr:
        for s in s_arr:
            
            src = "/research/harris/vivian/v1-models/saved-models/BN_V1_V1_Linear/s_f_sweep/" + "s_" + str(s) + "_f_" + str(f)
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
            

            string = str(s) + " " + str(f) + " " + str(avg_loss) + " " + "+/-" + str(loss_err) + " " + str(avg_acc) + " " + "+/-" + str(acc_err)
            file.write(string)
            file.write("\n")
            
           
            

    
    
    
    