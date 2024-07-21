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
from copy_align import NewAlignment, ANewAlignment
from align import Alignment

def realign(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            realign(m1)
        if isinstance(m1, Alignment):
            setattr(model, n1, NewAlignment(m1.rank, m1.rank))


def factconv(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            factconv(m1)
        if isinstance(m1, ResamplingDoubleFactConv2d):
            module=m1
            new_module = FactConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['resampling_weight']
            new_sd['tri1_vec'] = old_sd['tri1_vec']
            new_sd['tri2_vec'] = old_sd['tri2_vec']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            new_module.load_state_dict(new_sd)
            setattr(model, n1, new_module)
        if isinstance(m1, Alignment) or isinstance(m1, NewAlignment):
            new_module = ANewAlignment(m1.rank, m1.rank)
            new_module.alignment = m1.alignment.detach()
            setattr(model, n1, new_module)


def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1, ResamplingDoubleFactConv2d):
            m1.resample()


def load_model(args, model):
    #src="../saved-models/Long_Cifar_ResNets/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_rainbow_cifar/"
    run_name = "{}_batchsize_{}_rank_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(args.net,
            args.batchsize, args.rank,
            #1 if args.width == 0.125 else args.double, args.resample,
            args.double, args.resample,
              args.width, args.seed, args.num_epochs)
    sd = torch.load(src+run_name+"/model.pt")
    #for key in sd.keys():
    #    if "resampling_weight" in key:
    #        temp = sd[key.replace("resampling_weight", "weight")]
    #        #sd[key.replace("resampling_weight", "weight")] = sd[key]
    #        sd[key] = temp
    print(sd.keys())
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
batch=1000
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, shuffle=True,drop_last=True,  num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch, shuffle=True,drop_last=True, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = define_models(args)
run_name = "optimize_{}_{}_batchsize_{}_rank_{}_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(
        args.optimize,args.net, args.batchsize, args.rank, args.double,
        args.resample, args.width, args.seed, args.num_epochs)
#run_name = "width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width,
#        args.optimize, args.statistics, args.seed)
#
#run_name = "width_{}_seed_{}_pretrained".format(args.width, args.seed)
#run_name = "width_{}_seed_{}_pretrained".format(args.width, args.seed)

run_name = "explore_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)
print("Args.net: ", args.net)
print("Net: ", net)
set_seeds(0)
#set_seeds(args.seed)
if args.double:
    net = net.double()
net = net.to(device)
wandb_dir = "../../wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

run = wandb.init(project="FactConv", entity="muawizc", config=args,
        group="final_loading_align_resnet_cifar", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)
sd = load_model(args, net)
net.load_state_dict(sd)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.linear.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#        T_max=args.num_epochs)
args.resample=0
logger = {}
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    if args.statistics:
        net.train()
    else:
        net.eval()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.double:
            inputs = inputs.double()
        if args.resample:
            resample(net)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if args.optimize:
            loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    logger["train_accuracy"] = 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    state_dict = net.state_dict()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.double:
                inputs = inputs.double()
            if args.resample:
                resample(net)
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
    logger["accuracy"] = acc
    net.load_state_dict(state_dict)
    #if acc > best_acc:
    #    print('Saving..')
    #    state = {
    #        'net': net.state_dict(),
    #        'acc': acc,
    #        'epoch': epoch,
    #    }
    #    save_model(args, net)
    #    best_acc = acc
recorder = {}
#set_seeds(args.seed)
set_seeds(0)
net.cuda()
#resample(net)
print(net)
test(0)
recorder['epoch_0'] = logger['accuracy']
if args.statistics:
    realign(net)
for epoch in range(0, 10):
    train(epoch)
    test(epoch)
    recorder['epoch_{}'.format(epoch+1)] = logger['accuracy']
    recorder['epoch_train_{}'.format(epoch+1)] = logger['train_accuracy']
    if epoch == 5:
        factconv(net)
        net.cuda()
run.log(recorder)
