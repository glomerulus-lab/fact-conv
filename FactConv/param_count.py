'''Train CIFAR10 with PyTorch.'''
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
import wandb
from distutils.util import strtobool
from models import define_models

def save_model(args, model):
    src= "../saved-models/ResNets/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir+ "/model.pt")
    torch.save(args, model_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', type=str, default='resnet18', help="which model to use")
parser.add_argument('--num_epochs', type=int, default=200, help='number of trainepochs')
parser.add_argument('--name', type=str, default='ResNet', 
                        help='filename for saved model')
parser.add_argument('--seed', default=0, type=int, help='seed to use')
parser.add_argument('--width', type=float, default=1, help='resnet width scale factor')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

wandb_dir = "../../wandb"
names = []
names = ['resnet18', 'fact_resnet18', 'fact_us_resnet18', 'fact_uc_resnet18','fact_usuc_resnet18']
for name in ["fact_diag_resnet18", "fact_diag_us_resnet18",
        "fact_diag_uc_resnet18", "fact_diag_us_uc_resnet18",
        'resnet18',
        'fact_resnet18', 'fact_us_resnet18', 'fact_uc_resnet18','fact_us_uc_resnet18']:
    for width in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
        args.net = name
        args.width = width
        run_name = "{}_width_{}".format(args.net, args.width)
        run = wandb.init(project="FactConv", config=args,
                group="paramcount_presentation", name=run_name, dir=wandb_dir)
        # Model
        net = define_models(args)
        run.log({"params":sum(p.numel() for p in net.parameters() if p.requires_grad)})
        print(args.net, args.width)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        run.finish()
