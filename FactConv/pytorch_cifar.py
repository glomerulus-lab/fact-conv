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
    #src= "../saved-models/ResNets/"
    src="/home/mila/v/vivian.white/scratch/v1-models/saved-models/low-rank/"
    model_dir =  src + args.net + "-seed" + str(args.seed)
    print("Model dir: ", model_dir)
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
parser.add_argument('--spatial_k', type=float, default=1, help='%spatial low-rank')
parser.add_argument('--channel_k', type=float, default=1, help='%channel low-rank')

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

net = define_models(args)
run_name = "{}_width_{}_seed_{}".format(args.net, args.width, args.seed)
print("Args.net: ", args.net)
print("Net: ", net)

set_seeds(args.seed)

net = net.to(device)
wandb_dir = "/home/mila/v/vivian.white/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

print("Num Learnable Params: ", sum(p.numel() for p in net.parameters() if
    p.requires_grad))

run = wandb.init(project="factconv", config=args, group="covtest", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,\
        T_max=args.num_epochs)

# Training
def train(epoch):
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

def test(epoch):
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
    run.log({"accuracy":acc})
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_model(args, net)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
args.name += "final"
save_model(args, net)
