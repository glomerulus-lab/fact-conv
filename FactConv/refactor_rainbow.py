'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import wandb
import numpy as np

import time 
import os
import argparse
import copy
import gc
from distutils.util import strtobool

from pytorch_cifar_utils import progress_bar, set_seeds
from models.resnet import ResNet18
from conv_modules import FactConv2d
from models.function_utils import replace_layers_factconv2d,\
replace_layers_scale, replace_layers_fact_with_conv, turn_off_backbone_grad, \
recurse_preorder
from rainbow import RainbowSampler

def save_model(args, model):
    src = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/refactoring/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    
    torch.save(model.state_dict(), model_dir+ "/model.pt")
    torch.save(args, model_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--aca', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Activation Cross-Covariance Alignment')
parser.add_argument('--wa', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Weight alignment True=Yes False=No')
parser.add_argument('--in_wa', type=lambda x: bool(strtobool(x)), 
                        default=True, help='input=True output=False')
parser.add_argument('--fact', type=lambda x: bool(strtobool(x)), 
                        default=True, help='FactNet True or False')
parser.add_argument('--width', default=0.125, type=float, help='width')
parser.add_argument('--sampling', type=str, default='structured_alignment',
        choices=['structured_alignment', 'cc_specification'], help="which sampling to use")

args = parser.parse_args()

if int(args.width) == args.width:
    args.width = int(args.width)

print("Sampling: {} Width: {} Fact: {} ACA: {} WA: {} In_WA: {}".format(args.sampling,
    args.width, args.fact, args.aca, args.wa, args.in_wa))

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
    trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')


criterion = nn.CrossEntropyLoss()

def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    print("accuracy:", acc)
    return acc, test_loss

logger ={'width':args.width}#, }
set_seeds(args.seed)
for i in range(0, 1):
    net=ResNet18()
    replace_layers_scale(net, args.width)
    if args.fact:
        replace_layers_factconv2d(net)
    
    
    if args.fact:
        sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/fact_model.pt".format(args.width))
    elif not args.fact:
        sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/conv_model.pt".format(args.width))
    net.load_state_dict(sd)
    net.to(device)
    print(net)

    set_seeds(i)
    print("testing Res{}Net18 with width of {}".format("Fact" if args.fact else "Conv", args.width))
    pretrained_acc, og_loss = test(0, net)


    s=time.time()
    args.seed = i
    rainbow = RainbowSampler(net, trainloader, args.seed, args.sampling, args.wa, args.in_wa, args.aca, device)
    rainbow_net = rainbow.sample()
    rainbow_net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = rainbow_net(inputs)
    print("TOTAL TIME:", time.time()-s)
    turn_off_backbone_grad(rainbow_net)
    optimizer = optim.SGD(filter(lambda param: param.requires_grad, rainbow_net.parameters()), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    print("testing {} sampling at width {}".format(args.sampling, args.width))
    rainbow_net.eval()
    
    print(rainbow_net)
    
    sampled_acc, sampled_loss = test(0, rainbow_net)
    save_model(args, rainbow_net)
    accs = []
    test_losses= []
    print("training classifier head of {} sampled model for {} epochs".format(args.sampling, args.epochs))
    for j in range(0, args.epochs):
        rainbow_net.train()
        train(j, rainbow_net)
        rainbow_net.eval()
        acc, loss_test =test(j, rainbow_net)
        test_losses.append(loss_test)
        accs.append(acc)

    new_logger ={"sampled_acc_{}".format(i): sampled_acc,"pretrained_acc_{}".format(i):
            pretrained_acc, "og_loss_{}".format(i): og_loss,
        "first_epoch_acc_{}".format(i):accs[0], "third_epoch_acc_{}".format(i): accs[2],
        "tenth_epoch_acc_{}".format(i):accs[args.epochs-1], 
        "sampled_loss_{}".format(i):sampled_loss,
        "first_epoch_loss_{}".format(i):test_losses[0], "third_epoch_loss_{}".format(i): test_losses[2],
        "tenth_epoch_loss_{}".format(i):test_losses[args.epochs-1]}
    logger = {**logger, **new_logger}

wandb_dir = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
#group_string = "refactor"
group_string = "IGNOREvariance_runs"

#run_name = "refactor"
run_name= "width_{}_sampling_{}_fact_{}_ACA_{}_WA_{}_inWA_{}".format(args.width, args.sampling, args.fact, args.aca, args.wa, args.in_wa)
args.name = run_name
run = wandb.init(project="random_project", config=args,
        group=group_string, name=run_name, dir=wandb_dir)
run.log(logger)
