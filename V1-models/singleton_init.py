# This script initializes a conv net with a specified seed and saves it.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from pytorch_cifar_utils import progress_bar, set_seeds
from hooks import wandb_forwards_hook, wandb_backwards_hook
import wandb
from distutils.util import strtobool
from resnet import ResNet18

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--seed', type=int, default=0, help='init seed')
parser.add_argument('--model_name', type=str, default='conv_model_init',
        help='filename for saved model')
args = parser.parse_args()

save_dir = "/home/mila/v/vivian.white/scratch/v1-models/saved-models/singleton/"
os.makedirs(save_dir, exist_ok=True)
set_seeds(args.seed)
conv_model = ResNet18()
torch.save(conv_model.state_dict(), '{}/{}.pt'.format(save_dir, args.model_name))
print("Saved model with seed {} to {}".format(args.seed, save_dir))
