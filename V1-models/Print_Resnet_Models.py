#Model w/ weights initialized according to V1 receptive fields

import torch
import torch.optim
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets
from datetime import datetime
import os
from distutils.util import strtobool
import argparse
import V1_models
import LC_models
from test_models_safety import Replacenet_V1_CIFAR10, Factnet_V1_CIFAR10
import numpy as np
import random

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


    
def save_model(args, model, loss, accuracy):
    src = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = model_dir + "/trial_" + str(args.trial)
    os.makedirs(trial_dir, exist_ok=True)
    os.chdir(trial_dir)
    
    torch.save(loss, trial_dir+"/loss.pt")
    torch.save(accuracy, trial_dir+ "/accuracy.pt")
    torch.save(model.state_dict(), trial_dir+ "/model.pt")
    torch.save(args, trial_dir+ "/args.pt")


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    parser.add_argument('--name', type=str, default='Resnet_V1', 
                        help='filename for saved model')
    parser.add_argument('--trial', type=int, default=1, help='trial number')
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), 
                        default=False, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, help="which device to use (0 or 1)")
    parser.add_argument('--freeze_spatial', dest='freeze_spatial', 
                        type=lambda x: bool(strtobool(x)), default=True, 
                        help="freeze spatial filters for LearnableCov models")
    parser.add_argument('--freeze_channel', dest='freeze_channel', 
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="freeze channels for LearnableCov models")
    parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                        help="initialization for spatial filters for LearnableCov models")
    parser.add_argument('--net', type=str, default='fact', choices=['fact', 'replace'], 
                        help="which network to use for this test")
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    start = datetime.now()
    set_seeds(args.seed)
    print("Printing Resnet model structure with defined FactConv2d:")
    print(Factnet_V1_CIFAR10(args.s, args.f, args.scale, args.bias,
        args.freeze_spatial, args.freeze_channel,
        args.spatial_init).to(device))
    print("Printing Resnet model structure with replaced-by FactConv2d:")
    print(Replacenet_V1_CIFAR10(args.s, args.f, args.scale, args.bias,
        args.freeze_spatial, args.freeze_channel,
        args.spatial_init).to(device))


