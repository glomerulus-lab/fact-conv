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
from pytorch_cifar_utils import progress_bar, set_seeds, accuracy, AverageMeter
import wandb
from distutils.util import strtobool
from models import define_models
from conv_modules import ResamplingDoubleFactConv2d

def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1, nn.Conv2d):# ResamplingDoubleFactConv2d):
            m1.resample()

def save_model(args, model):
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_new_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/retry_recent_new_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/top3_recent_new_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/final_rainbows/"
    #src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/gmm_rainbow_cifar/"
    run_name = "{}_batchsize_{}_rank_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(args.net,
            args.batchsize, args.rank,
            args.double, args.resample, args.width, args.seed, args.num_epochs)
    model_dir =  src + run_name
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
parser.add_argument('--double', default=0, type=int, help='seed to use')
parser.add_argument('--gmm', default=0, type=int, help='seed to use')
parser.add_argument('--channel_k', default=512, type=int, help='seed to use')
parser.add_argument('--resample', default=0, type=int, help='seed to use')
parser.add_argument('--batchsize', default=256, type=int, help='seed to use')
parser.add_argument('--rank', default=200, type=int, help='seed to use')

parser.add_argument('--bias', default=0, type=int, help='seed to use')

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
    trainset, batch_size=args.batchsize, shuffle=True,drop_last=True,  num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batchsize, shuffle=True,drop_last=True, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:3, 6:5, 7:6, 8:7, 9:8}

# Model
print('==> Building model..')

net = define_models(args)
run_name = "{}_batchsize_{}_rank_{}_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(args.net, args.batchsize, args.rank, args.double,
        args.resample, args.width, args.seed, args.num_epochs)
print("Args.net: ", args.net)
print("Net: ", net)
set_seeds(args.seed)
if args.double:
    net = net.double()
net = net.to(device)
wandb_dir = "../../wandb"
#TODO: VIVIAN REMOVE REVERT BACK TO ICML DAYS!!!
#TODO: PLS NO MORE CHDIR Q_Q
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

run = wandb.init(project="FactConv", entity="muawizc", config=args,
        group="testing_saving_align_resnet_cifar", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=args.num_epochs)

logger = {}
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    top1 = AverageMeter("Acc@1")
    top2 = AverageMeter("Acc@2")
    top3 = AverageMeter("Acc@3")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.double:
            inputs = inputs.double()
        if args.resample:
            resample(net)

        optimizer.zero_grad()
        outputs = net(inputs)
        if "pre_bn_alt_aligned_resnet18" in args.net:
            targets = torch.cat([targets, targets], dim=0)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, acc2, acc3 = accuracy(outputs, targets, topk=(1, 2, 3))
        
        top1.update(acc1[0], outputs.shape[0])
        top2.update(acc2[0], outputs.shape[0])
        top3.update(acc3[0], outputs.shape[0])

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%%  (%d/%d), top1: %.3f%%, top2: %.3f%%, top3: %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct,
                         total, top1.avg, top2.avg, top3.avg))
    
    logger["train_accuracy"] = 100.*correct/total
    logger['top1_train'] = top1.avg
    logger['top2_train'] = top2.avg
    logger['top3_train'] = top3.avg


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    top1 = AverageMeter("Acc@1")
    top2 = AverageMeter("Acc@2")
    top3 = AverageMeter("Acc@3")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.double:
                inputs = inputs.double()
            if args.resample:
                resample(net)
            outputs = net(inputs)#[:inputs.shape[0]]
            if "pre_bn_alt_aligned_resnet18" in args.net:
                targets = torch.cat([targets, targets], dim=0)
            loss = criterion(outputs, targets)

            acc1, acc2, acc3 = accuracy(outputs, targets, topk=(1, 2, 3))
 
            top1.update(acc1[0], outputs.shape[0])
            top2.update(acc2[0], outputs.shape[0])
            top3.update(acc3[0], outputs.shape[0])

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d), top1: %.3f%%, top2: %.3f%%, top3: %.3f%%'
                         % (test_loss/(batch_idx+1), 100.*correct/total,
                             correct, total, top1.avg, top2.avg, top3.avg))

    # Save checkpoint.
    acc = 100.*correct/total
    logger["accuracy"] = acc
    logger['top1_test'] = top1.avg
    logger['top2_test'] = top2.avg
    logger['top3_test'] = top3.avg
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
    run.log(logger)#
    scheduler.step()
args.name += "final"
save_model(args, net)
