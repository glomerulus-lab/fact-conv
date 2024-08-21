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
from models.attention import CrossAttentionNetwork

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

parser.add_argument('--fixed', default=0, type=int, help='seed to use')
parser.add_argument('--q', default=0, type=int, help='seed to use')
parser.add_argument('--k', default=0, type=int, help='seed to use')
parser.add_argument('--v', default=0, type=int, help='seed to use')
parser.add_argument('--num_samples', default=10, type=int, help='seed to use')
parser.add_argument('--pre_processing', default="", type=str, help='seed to use')
parser.add_argument('--post_processing', default="linear", type=str, help='seed to use')
parser.add_argument('--num_heads', default=1, type=int, help='seed to use')

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
import numpy as np
targets_data = np.array(testset.targets)
indices = np.arange(0, len(testset.targets))
np.random.shuffle(indices)
testset.targets = targets_data[indices].tolist()
testset.data = testset.data[indices]

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

run_name = "{}_num_samples_{}_fixed_{}_q_{}_k_{}_v_{}_preprocessing_{}_postprocessing_{}_num_heads".format(args.num_samples, args.fixed, args.q, args.k, args.v, args.pre_processing, args.post_processing, args.num_heads)
run = wandb.init(project="FactConv", entity="muawizc", config=args,
        group="final_multihead_attn_ensemble", name=run_name, dir=wandb_dir)
sd = load_model(args, net)
net.load_state_dict(sd)

criterion = nn.CrossEntropyLoss()
src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/rainbow_cifar_ensemble_cast/"
run_name = "{}_batchsize_{}_rank_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(args.net,
        args.batchsize, args.rank,
        args.double, 0, args.width, args.seed, args.num_epochs)

model_dir =  src + run_name
os.makedirs(model_dir, exist_ok=True)

print(model_dir)
nets=[]
if args.fixed:
    copy_net = copy.deepcopy(net)
    copy_net.cuda()
    for i in range(0, 10):
        print(i)
        sd = torch.load(model_dir+ "/model_{}.pt".format(i))
        copy_net.cuda()
        copy_net.load_state_dict(sd)
        copied_net = copy.deepcopy(copy_net)
        copied_net.linear = nn.Identity()
        nets.append(copied_net)
    #copy_net.cpu()
    nets = nn.ModuleList(nets)

print("NETS COPIED")
##TODO: CHANGE
attn = CrossAttentionNetwork(net, args.num_samples, nets=nets, q=args.q, k=args.k, v=args.v,
        pre_processing=args.pre_processing,
        post_processing=args.post_processing,
        num_heads=args.num_heads)
attn.cuda()

optimizer = optim.SGD(list(list(attn.classifier.parameters())+
    list(attn.attn.parameters())), lr=0.1, momentum=0.9,
        weight_decay=5e-4)
args.resample=0
logger = {}


net.cuda()

# Training
def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.eval()
    #[net.eval() for net in nets]
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if batch_idx==0:
            print(outputs.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    logger["train_accuracy"] = 100.*correct/total


def test(epoch, net):
    global best_acc
    net.eval()
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
    acc = 100.*correct/total
    logger["accuracy"] = acc


set_seeds(args.seed)
print(attn)
accs = []
for i in range(0, 5):
    set_seeds(i)
    train(i, attn)
    set_seeds(args.seed)
    test(i, attn)
    accs.append(logger['accuracy'])
    run.log(logger)
print(np.mean(np.array(accs)))
print(stats.sem(np.array(accs)))
run.finish()
