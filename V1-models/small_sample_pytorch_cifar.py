'''Train CIFAR10 with PyTorch.'''
import torch
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
import numpy as np
from numpy.random import RandomState
    
def save_model(args, model, seed):
    src  = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/small_sample_CIFAR10_pytorch/seed_{}".format(seed)
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = model_dir + "/trial_" + str(1)
    os.makedirs(trial_dir, exist_ok=True)
    os.chdir(trial_dir)
    
    torch.save(model.state_dict(), trial_dir+ "/model.pt")
    torch.save(args, trial_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', type=str, default='post', choices=['post', 'pre'], help="which convmodule to use")
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
parser.add_argument('--name', type=str, default='TESTING_WATCH_Resnet_V1', 
                        help='filename for saved model')
parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), 
                        default=False, help='bias=True or False')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_accs = []
for seed in range(0, 10):

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
    
    cifar_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    prng = RandomState(seed)
    random_permute = prng.permutation(np.arange(0, 5000))[0:50]
    indx = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute] for classe in range(0, 10)])
    
    cifar_data.data, cifar_data.targets = cifar_data.data[indx], list(np.array(cifar_data.targets)[indx])
     
    
    trainloader = torch.utils.data.DataLoader(
        cifar_data, batch_size=128, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=4)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Model
    print('==> Building model..')
    set_seeds(0)
    net = PreExp(args.s, args.f, args.scale, args.bias, args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)
    set_seeds(0)
    
    net = net.to(device)
    store_rms = {}
    #for name, module in net.named_modules():
    #    if isinstance(module, nn.Conv2d):
    #        ## simple module
    #        module.register_forward_hook(wandb_forwards_hook("{}".format(name), store_rms))
    #        if module.tri1_vec.requires_grad:
    #            module.tri1_vec.register_hook(wandb_backwards_hook("{} Tri1_vec Grad".format(name), store_rms))
    #        if module.tri2_vec.requires_grad:
    #            module.tri2_vec.register_hook(wandb_backwards_hook("{} Tri2_vec Grad".format(name), store_rms))
    
    
    
    #if device == 'cuda':
        #net = torch.nn.DataParallel(net)
        #cudnn.benchmark = True
    wandb_dir = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/wandb"
    os.makedirs(wandb_dir, exist_ok=True)
    os.chdir(wandb_dir)
    run_name = ""
    if args.freeze_spatial:
        run_name += "US"
    else:
        run_name += "LS"
    
    if args.freeze_channel:
        run_name += "UC"
    else:
        run_name += "LC"
    
    
    run = wandb.init(project="random_project", config=args,
            group="testing_pytorch_cifar_experiments_small_sample", name=run_name, dir=wandb_dir)
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
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    import time
    
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
            torch.cuda.synchronize()
            loss = criterion(outputs, targets)
            s1=time.time()
            loss.backward()
    
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #run.log(store_rms)
    
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
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            #if not os.path.isdir('checkpoint'):
            #    os.mkdir('checkpoint')
            #torch.save(state, './checkpoint/ckpt.pth')
            save_model(args, net, seed)
            best_acc = acc
    
    
    for epoch in range(start_epoch, start_epoch+200):#00
        train(epoch)
        test(epoch)
        scheduler.step()
    
    args.name += "final"
    save_model(args, net, seed)
    best_accs.append(best_acc)
    print("Finished seed {}".format(seed), "Best_accuracy: {}".format(best_acc))
print(best_accs)
def print_results(a):
  a = np.array(a)
  print(len(a))
  plusminus = u'\u00b1'
  print(a)
  print(str(round(np.mean(a), 2)) + " " + plusminus + " " + str(round(np.std(a)/np.sqrt(len(a)), 4)))
print_results(best_accs)
#
#        with profile(activities=[ProfilerActivity.CPU,
#            ProfilerActivity.CUDA],with_stack=True) as prof:
#            s1=time.time()
#            loss.backward()
#            print(time.time()-s1)
#        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

