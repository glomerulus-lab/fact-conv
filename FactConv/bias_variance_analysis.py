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
from conv_modules import ResamplingDoubleFactConv2d, FactConv2d
from copy_align import NewAlignment
from align import Alignment
import math
import copy
from scipy import stats

def realign(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            realign(m1)
        if isinstance(m1, Alignment) or isinstance(m1, NewAlignment):
            setattr(model, n1, NewAlignment(m1.rank, m1.rank))

def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1,nn.Conv2d):
            m1.resample()


def load_model(args, model):
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_rainbow_cifar/"
    #src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/retry_recent_new_rainbow_cifar/"
    run_name = "{}_batchsize_{}_rank_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(args.net,
            args.batchsize, args.rank,
            args.double, args.resample,
              args.width, 0, args.num_epochs)
    sd = torch.load(src+run_name+"/model.pt")
    return sd

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', type=str, default='resnet18', help="which model to use")
parser.add_argument('--num_epochs', type=int, default=200, help='number of trainepochs')
parser.add_argument('--name', type=str, default='ResNet', 
                        help='filename for saved model')
parser.add_argument('--seed', default=0, type=int, help='seed to use')
parser.add_argument('--double', default=0, type=int, help='seed to use')
parser.add_argument('--optimize', default=0, type=int, help='seed to use')
parser.add_argument('--statistics', default=0, type=int, help='seed to use')
parser.add_argument('--resample', default=0, type=int, help='seed to use')
parser.add_argument('--batchsize', default=256, type=int, help='seed to use')
parser.add_argument('--rank', default=200, type=int, help='seed to use')
parser.add_argument('--bias', default=0, type=int, help='seed to use')
parser.add_argument('--width', type=float, default=1, help='resnet width scale factor')
parser.add_argument('--gmm', default=0, type=int, help='seed to use')
parser.add_argument('--t', default=0, type=float, help='seed to use')

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
batch=1000
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, shuffle=True,drop_last=True,  num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch, shuffle=False,drop_last=True, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
print(len(testset.targets))
print(testset.data.shape)
import numpy as np
print(testset.targets)
targets_data = np.array(testset.targets)
indices = np.arange(0, len(testset.targets))
print(indices)
np.random.shuffle(indices)
print(indices)
testset.targets = targets_data[indices].tolist()
testset.data = testset.data[indices]
print(testset.targets)
print(testset.data.shape)


# Model
print('==> Building model..')

net = define_models(args)
run_name = "optimize_{}_{}_batchsize_{}_rank_{}_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(
        args.optimize,args.net, args.batchsize, args.rank, args.double,
        args.resample, args.width, args.seed, args.num_epochs)

run_name = "explore_corrected_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)
run_name = "no_bias_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)

run_name = "true_learn_bias_10_epochs_rainbow_width_{}_seed_{}".format(args.width, args.seed)
run_name = "bias_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)
run_name = "conv_width_{}_seed_{}".format(args.width, args.seed)

run_name = "{}_width_{}_seed_{}".format(args.name, args.width, args.seed)
print("Args.net: ", args.net)
set_seeds(0)
if args.double:
    net = net.double()
net = net.to(device)
print(net)
wandb_dir = "../../wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

run = wandb.init(project="FactConv", entity="muawizc", config=args,
        group="testing_bias_variance_analysis", name=run_name, dir=wandb_dir)
sd = load_model(args, net)
net.load_state_dict(sd)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.linear.parameters(), lr=0.1, momentum=0.9,
        weight_decay=5e-4)
args.resample=0
logger = {}
net.cpu()
nets = [copy.deepcopy(net) for i in range(0, 5)]
print("NETS COPIED")
[resample(net_i) for net_i in nets]
net.cuda()
print("DONE SEEDING")
# Training
def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    if args.statistics:
        net.train()
    else:
        net.eval()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.double:
                inputs = inputs.double()

            #optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #loss.backward()

            #if args.optimize:
            #    optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    logger["train_accuracy"] = 100.*correct/total

def test(epoch, net, net_c, data_dict):
    global best_acc
    net.eval()
    state_dict = net.state_dict()
    test_loss = 0
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.double:
                inputs = inputs.double()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            for i, output in enumerate(outputs):
                _, predicted = output.unsqueeze(0).max(1)
                data_dict["{}_epoch_{}_net_{}_prediction".format(epoch,net_c, i+counter)] = predicted.cpu().item()
                data_dict["{}_epoch_{}_net_{}_logits".format(epoch,net_c, i+counter)] = np.array(output.cpu()).tolist()
                data_dict["{}_epoch_{}_net_{}_target".format(epoch,net_c,i+counter)] = (targets[i].cpu()).item()
            counter += outputs.shape[0]


    acc = 100.*correct/total
    logger["accuracy"] = acc
    net.load_state_dict(state_dict)

data_dict = {}
recorder = {}
set_seeds(args.seed)
net.cuda()
print(net)
set_seeds(args.seed)
#test(0, net)
#set_seeds(args.seed)
#test(0, net)
set_seeds(args.seed)
accs = []
#for net_c, net in enumerate(nets):
#    net.cuda()
#    set_seeds(args.seed)
#    test(0, net, net_c, data_dict)
#    accs.append(logger['accuracy'])
#    net.cpu()
#print(data_dict.keys())
for net_c in range(0, 100):
    net.cuda()
    set_seeds(net_c)
    resample(net)
    set_seeds(args.seed)
    realign(net)
    set_seeds(args.seed)
    for k in range(0, 2):
        train(k, net)
    set_seeds(args.seed)
    test(0, net, net_c, data_dict)
    accs.append(logger['accuracy'])
    #net.cpu()
print(np.mean(np.array(accs)))
print(stats.sem(np.array(accs)))

import pickle 

os.chdir("/home/mila/m/muawiz.chaudhary/post_thesis_work/refactoring/v1-models/FactConv")
with open('saved_dictionary_adapted_1000.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
                
#with open('saved_dictionary_10.pkl', 'rb') as f:
#    loaded_dict = pickle.load(f)
#    print(loaded_dict.keys)
#    print("RELOADED")

#run.log(data_dict)
#recorder['epoch_0'] = logger['accuracy']
#if args.statistics:
#    realign(net)
#for epoch in range(0, 5):
#    train(epoch)
#    test(epoch+1)
#    #recorder['epoch_{}'.format(epoch+1)] = logger['accuracy']
#    #recorder['epoch_train_{}'.format(epoch+1)] = logger['train_accuracy']
#run.log(recorder)
