''')Train CIFAR10 with PyTorch.'''
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from pytorch_cifar_utils import progress_bar, set_seeds

from test_models_safety import PostExp, PreExp
from hooks import wandb_forwards_hook, wandb_backwards_hook

import wandb

from distutils.util import strtobool

from resnet import ResNet18
from vgg import VGG

from test_models_safety import PostExp, PreExp

#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True

def save_model(args, fact1, fact2):
    src="/home/mila/v/vivian.white/scratch/v1-models/saved-models/twins/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    #trial_dir = model_dir + "/trial_" + str(1)
    #os.makedirs(trial_dir, exist_ok=True)
    #os.chdir(trial_dir)
    trial_dir = model_dir 
    torch.save(fact1.state_dict(), trial_dir+ "/fact1.pt")
    torch.save(fact2.state_dict(), trial_dir+ "/fact2.pt")
    torch.save(args, trial_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', type=str, default='vgg', choices=['vgg', 'vggbn',
    'resnet', 'factnetv1', 'factnetdefault', 'vggfact', 'vggbnfact'], help="which convmodule to use")
parser.add_argument('--freeze_spatial', dest='freeze_spatial', 
                    type=lambda x: bool(strtobool(x)), default=True, 
                    help="freeze spatial filters for LearnableCov models")
parser.add_argument('--freeze_channel', dest='freeze_channel', 
                    type=lambda x: bool(strtobool(x)), default=False,
                    help="freeze channels for LearnableCov models")
parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                    help="initialization for spatial filters for LearnableCov models")
parser.add_argument('--s', type=int, default=2, help='V1 size')
parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
parser.add_argument('--scale', type=int, default=1, help='V1 scale')
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), 
                        default=False, help='bias=True or False')
parser.add_argument('--width_scale', type=float, default=1, 
                    help='width scale amount')

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

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#
#
#
from ConvModules import FactConv2dPreExp
from torchvision.models.resnet import resnet18


def simclr_replace_layers(model):
    for n, module in model.named_children():
        if n == 'conv1':
            new_module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                bias=False)
            setattr(model, n, new_module)
            print("Replaced first conv")
        if isinstance(module, nn.MaxPool2d):
            new_module = nn.Identity()
            setattr(model, n, new_module)
            print("REplaced maxpool")


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

def replace_layers_agnostic(model, scale=1):
    prev_out_ch = 0
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_agnostic(module,scale)
        if isinstance(module, nn.Conv2d):
            if module.in_channels == 3:
                in_scale = 1 
            else:
                in_scale = scale
            ## simple module
            new_module = nn.Conv2d(
                    in_channels=int(module.in_channels*in_scale),
                    out_channels=int(module.out_channels*scale),
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    groups = module.groups,
                    bias=True if module.bias is not None else False)
            setattr(model, n, new_module)
            prev_out_ch = new_module.out_channels
        if isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(prev_out_ch)
            setattr(model, n, new_module)
        if isinstance(module, nn.Linear):
            new_module = nn.Linear(int(512 * scale), 10)
            setattr(model, n, new_module)
set_seeds(0)
if args.net == "vgg":
    net=VGG("VGG11", False)
    conv_net = VGG("VGG11", False)
    run_name = "OGVGG"
elif args.net == "vggbn":
    net=VGG("VGG11", True)
    conv_net = VGG("VGG11", True)
    run_name = "OGVGGBN"
elif args.net == "resnet":
    # net = ResNet18()
    conv_net = resnet18()
    #simclr_replace_layers(conv_net)
    replace_layers_agnostic(conv_net, args.width_scale)
    run_name = "Width-Scaled Torchvision Resnet"

conv1 = resnet18().to(device)
fact1 = copy.deepcopy(conv1)
replace_layers_keep_weight(fact1)
fact1 = fact1.to(device)

set_seeds(9)
conv2 = resnet18().to(device)
fact2 = copy.deepcopy(conv2)
replace_layers_keep_weight(fact2)
fact2 = fact2.to(device)
run_name = "Different Init Facts"
print("Built models")
old_name = args.name
args.name += "_init"
save_model(args, fact1, fact2) 
args.name = old_name
set_seeds(0)
#if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True
wandb_dir = "/home/mila/v/vivian.white/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
#run_name = "OGVGG"

run = wandb.init(project="random_project", config=args,
        group="pytorch_cifar_twins", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
fact1_optimizer = optim.SGD(fact1.parameters(),
            lr=args.lr, momentum=0.9, weight_decay=5e-4)
fact2_optimizer = optim.SGD(fact2.parameters(),
            lr=args.lr, momentum=0.9, weight_decay=5e-4)
fact1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fact1_optimizer, T_max=200)
fact2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fact2_optimizer, T_max=200)

keys = ['fact1_acc_train', 'fact2_acc_train', 'fact1_acc_test', 'fact2_acc_test']
logger = {key: None for key in keys}
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    fact1.train()
    fact2.train()

    fact1_train_loss = 0
    fact1_correct = 0
    fact2_train_loss = 0
    fact2_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        fact1_optimizer.zero_grad()
        fact2_optimizer.zero_grad()

        fact1_outputs = fact1(inputs)
        fact1_loss = criterion(fact1_outputs, targets)

        fact2_outputs = fact2(inputs)
        fact2_loss = criterion(fact2_outputs, targets)

        fact1_loss.backward()
        fact1_optimizer.step()

        fact2_loss.backward()
        fact2_optimizer.step()

        fact1_train_loss += fact1_loss.item()
        _, fact1_predicted = fact1_outputs.max(1)
        total += targets.size(0)
        fact1_correct += fact1_predicted.eq(targets).sum().item()

        fact2_train_loss += fact2_loss.item()
        _, fact2_predicted = fact2_outputs.max(1)
        fact2_correct += fact2_predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 
                'Fact1 Loss: %.3f | Fact1 Acc: %.3f%% (%d/%d) | Fact2 Loss: %.3f | Fact2 Acc: %.3f%% (%d/%d)' 
                     % (fact1_train_loss/(batch_idx+1),
                         100.*fact1_correct/total, fact1_correct, total,
                        fact2_train_loss/(batch_idx+1),
                         100.*fact2_correct/total, fact2_correct, total))
    logger['fact1_acc_train'] = 100*fact1_correct/total
    logger['fact2_acc_train'] = 100*fact2_correct/total

def test(epoch):
    global best_acc
    fact1.eval()
    fact1_test_loss = 0
    fact1_correct = 0
    total = 0

    fact2.eval()
    fact2_test_loss = 0
    fact2_correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            fact1_outputs = fact1(inputs)
            fact1_loss = criterion(fact1_outputs, targets)

            fact2_outputs = fact2(inputs)
            fact2_loss = criterion(fact2_outputs, targets)

            fact1_test_loss += fact1_loss.item()
            _, fact1_predicted = fact1_outputs.max(1)
            total += targets.size(0)
            fact1_correct += fact1_predicted.eq(targets).sum().item()

            fact2_test_loss += fact2_loss.item()
            _, fact2_predicted = fact2_outputs.max(1)
            fact2_correct += fact2_predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Fact1 Loss: %.3f | Fact1 Acc: %.3f%% (%d/%d) | Fact2 Loss: %.3f | Fact2 Acc: %.3f%% (%d/%d)'
                         % (fact1_test_loss/(batch_idx+1),
                             100.*fact1_correct/total, fact1_correct, total,
                         fact2_test_loss/(batch_idx+1),
                             100.*fact2_correct/total, fact2_correct, total))
            
        logger['fact1_acc_test'] = 100*fact1_correct/total
        logger['fact2_acc_test'] = 100*fact2_correct/total
    # Save checkpoint.
#    if fact_acc > fact_best_acc:
#        print('Saving..')
#        state = {
#            'fact_acc': fact_acc,
#            'conv_acc': conv_acc,
#            'epoch': epoch,
#        }
#        #if not os.path.isdir('checkpoint'):
#        #    os.mkdir('checkpoint')
#        #torch.save(state, './checkpoint/ckpt.pth')
#        fact_best_acc = fact_acc
#    if conv_acc > conv_best_acc:
#        print('Saving..')
#        state = {
#            'fact_acc': fact_acc,
#            'conv_acc': conv_acc,
#            'epoch': epoch,
#        }
#        #if not os.path.isdir('checkpoint'):
#        #    os.mkdir('checkpoint')
#        #torch.save(state, './checkpoint/ckpt.pth')
#        conv_best_acc = conv_acc


for epoch in range(start_epoch, start_epoch+200):#00
    train(epoch)
    test(epoch)
    fact1_scheduler.step()
    fact2_scheduler.step()
    args.name += "_epoch_{}".format(epoch)
    run.log(logger)
    save_model(args, fact1, fact2) # save every epoch
    args.name = old_name

args.name = old_name
args.name += "_final"
save_model(args, fact1, fact2)
