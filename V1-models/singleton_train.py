# This script loads a saved initialized conv model, keeps it as conv or
# changes to fact, and trains it with an optimization seed.
import copy
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


def save_model(args, model):
    src= "/home/mila/v/vivian.white/scratch/v1-models/saved-models/singleton/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    #trial_dir = model_dir + "/trial_" + str(1)
    #os.makedirs(trial_dir, exist_ok=True)
    #os.chdir(trial_dir)
    trial_dir = model_dir 
    torch.save(model.state_dict(), trial_dir+ "/model.pt")
    torch.save(args, trial_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', type=str, default='conv', choices=['conv','fact'],
                    help="which convmodule to use")
parser.add_argument('--freeze_spatial', dest='freeze_spatial', 
                    type=lambda x: bool(strtobool(x)), default=True, 
                    help="freeze spatial filters for LearnableCov models")
parser.add_argument('--freeze_channel', dest='freeze_channel', 
                    type=lambda x: bool(strtobool(x)), default=False,
                    help="freeze channels for LearnableCov models")
parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                    help="initialization for spatial filters for LearnableCov models")
parser.add_argument('--name', type=str, default='SingletonConv', 
                    help='filename for saved model')
parser.add_argument('--load_model', type=str, default='conv_model_init.pt',
                    help='filename for loaded model')
parser.add_argument('--seed', type=int, default=0, help='training seed')

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



from ConvModules import FactConv2dPreExp

def replace_layers_keep_weight(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_keep_weight(module)
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['weight']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            new_module.load_state_dict(new_sd)
            #new_module.tri1_vec = nn.Parameter(int(new_module.tri1_vec * scale))
            setattr(model, n, new_module)

save_dir\
= "/home/mila/v/vivian.white/scratch/v1-models/saved-models/singleton/"+args.load_model
print("Save dir: ", save_dir)
conv_model = ResNet18()
initial = torch.load(save_dir)
conv_model.load_state_dict(initial)
model = conv_model.to(device)
print("Model Loaded Successfully")
if args.net == 'fact':
    print("Creating FactNet")
    fact_model = copy.deepcopy(conv_model)
    replace_layers_keep_weight(fact_model)
    model = fact_model.to(device)


#run_name = "Singleton Convs"
run_name = args.name
old_name = args.name
#args.name += "_init"
#save_model(args, model) 
#args.name = old_name
set_seeds(args.seed)
wandb_dir = "/home/mila/v/vivian.white/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
run = wandb.init(project="random_project", config=args,
        group="pytorch_cifar_singleton", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
            lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

keys = ['acc_train','acc_test']
logger = {key: None for key in keys}



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' 
                     % (train_loss/(batch_idx+1),
                         100.*correct/total, correct, total))
    logger['acc_train'] = 100*correct/total

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |'
                         % (test_loss/(batch_idx+1),
                             100.*correct/total, correct, total))
            
        logger['acc_test'] = 100*correct/total


for epoch in range(start_epoch, start_epoch+200):#00
    train(epoch)
    test(epoch)
    scheduler.step()
    args.name += "_epoch_{}".format(epoch)
    run.log(logger)
    save_model(args, model) # save every epoch
    args.name = old_name

args.name = old_name
args.name += "_final"
save_model(args, model)
