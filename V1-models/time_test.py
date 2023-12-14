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

from pytorch_cifar_utils import progress_bar

import wandb

from distutils.util import strtobool

from resnet import ResNet18
import time

import numpy as np
import random

   
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
parser.add_argument('--config', type=int, default=1, help='Config')

args = parser.parse_args()
    
def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


if args.config == 1:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.benchmark = False
elif args.config == 2:
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
elif args.config == 3:
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = False
elif args.config == 4:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
elif args.config == 5:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.benchmark = True
elif args.config == 6:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


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
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=4)

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
set_seeds(0)

net = ResNet18()
set_seeds(0)
set_seeds(0)

net = net.to(device)
run_name = "Time Test Config {}".format(args.config)

run = wandb.init(project="not_random_project", config=args,
        group="pytorch_time_testing_cifar_10_finegrained", name=run_name)#, dir=wandb_dir)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
logger={}

def record_event(events, name, stream=None):
    event = torch.cuda.Event(enable_timing=True)
    event.record(stream=stream)
    
    # Add or append event to dict.
    if name in events: events[name].append(event)
    else:              events[name] = [event]


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    non_blocking = False
    events = {}

    trainloader_len = len(trainloader)

    record_event(events, 'iter_start')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        record_event(events, 'data_loaded')

        inputs  = inputs .to(device=device, non_blocking=non_blocking)
        targets = targets.to(device=device, non_blocking=non_blocking)
        record_event(events, 'data_uploaded')

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        record_event(events, 'forward_passed')

        loss.backward()
        optimizer.step()
        record_event(events, 'backward_passed')
        train_loss += loss.item()
        record_event(events, 'loss_downloaded')

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #progress_bar(batch_idx, trainloader_len, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


        record_event(events, 'other_stuff')
        record_event(events, 'iter_start')
    #
    # NOTE: elapsed_time() returns the time in units of milliseconds.
    # NOTE: We will ignore the first iteration's timings as they are effectively warmup.
    #
    N = batch_idx
    epoch_time = events['iter_start'][0].elapsed_time(events['iter_start'][-1]) / 1000.0
    data_load_time     = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['iter_start'][1:],
                                          events['data_loaded'][1:])
    )
    data_upload_time   = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['data_loaded'][1:],
                                          events['data_uploaded'][1:])
    )
    processing_time    = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['data_uploaded'][1:],
                                          events['forward_passed'][1:])
    )
    backwards_time    = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['forward_passed'][1:],
                                          events['backward_passed'][1:])
    )
    loss_download_time = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['backward_passed'][1:],
                                          events['loss_downloaded'][1:])
    )
    other_stuff_time   = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['loss_downloaded'][1:],
                                          events['other_stuff'][1:])
    )
    average_iter_time = 0.001 / N * sum(
            s.elapsed_time(e) for s, e in zip(events['iter_start'][1:-1],
                                          events['iter_start'][2:])
            )    
    logger["train_time_per_batch"] = average_iter_time
    logger["train_time"] = epoch_time
    logger["train_time_per_forward"] = processing_time
    logger["train_time_per_backward"] = backwards_time
    logger["train_time_latency"] = data_load_time + data_upload_time
    logger["train_time_dataload"] = data_load_time 
    logger["train_time_dataupload"] = data_upload_time 
    logger["train_time_loss"] = loss_download_time
    logger["train_time_other"] = other_stuff_time





def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    testloader_len = len(testloader)
    
    non_blocking = False
    events = {}

    with torch.no_grad():
        record_event(events, 'iter_start')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            record_event(events, 'data_loaded')
            
            inputs  = inputs .to(device=device, non_blocking=non_blocking)
            targets = targets.to(device=device, non_blocking=non_blocking)
            record_event(events, 'data_uploaded')
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            record_event(events, 'forward_passed')

            test_loss += loss.item()
            record_event(events, 'loss_downloaded')
            
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(
            #    batch_idx, testloader_len,
            #    f'Loss: {test_loss/(batch_idx+1):.3f} | '
            #    f'Acc: {100.*correct/total:.3f}% ({correct}/{total})'
            #)
            #
            record_event(events, 'other_stuff')
            record_event(events, 'iter_start')

    torch.cuda.synchronize()
    
    #
    # NOTE: elapsed_time() returns the time in units of milliseconds.
    # NOTE: We will ignore the first iteration's timings as they are effectively warmup.
    #
    N = batch_idx
    epoch_time = events['iter_start'][0].elapsed_time(events['iter_start'][-1]) / 1000.0
    data_load_time     = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['iter_start'][1:],
                                          events['data_loaded'][1:])
    )
    data_upload_time   = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['data_loaded'][1:],
                                          events['data_uploaded'][1:])
    )
    processing_time    = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['data_uploaded'][1:],
                                          events['forward_passed'][1:])
    )
    loss_download_time = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['forward_passed'][1:],
                                          events['loss_downloaded'][1:])
    )
    other_stuff_time   = 0.001 / N * sum(
        s.elapsed_time(e) for s, e in zip(events['loss_downloaded'][1:],
                                          events['other_stuff'][1:])
    )
    average_iter_time = 0.001 / N * sum(
            s.elapsed_time(e) for s, e in zip(events['iter_start'][1:-1],
                                          events['iter_start'][2:])
            )
    logger["eval_time_per_batch"] = average_iter_time
    logger["eval_time"] = epoch_time
    logger["eval_time_per_forward"] = processing_time
    logger["eval_time_latency"] = data_load_time + data_upload_time
    logger["eval_time_dataload"] = data_load_time 
    logger["eval_time_dataupload"] = data_upload_time 
    logger["eval_time_loss"] = loss_download_time
    logger["eval_time_other"] = other_stuff_time

    logger["eval_acc"] = 100.*correct/total

for epoch in range(start_epoch, start_epoch+200):#00
    train(epoch)
    test(epoch)
    scheduler.step()
    run.log(logger)
    logger = {}
args.name += "final"
