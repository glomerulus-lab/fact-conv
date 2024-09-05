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
from copy_align import NewAlignment, NewAltAlignment
from align import Alignment
from models.altaligned_resnet import Alignment as AltAlignment
import math

convert={64: 256, 128:512, 256:1024, 512:2048}
def realign(model, mode, mom):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            realign(m1, mode, mom)
        if isinstance(m1, Alignment):
            setattr(model, n1, NewAlignment(m1.rank, m1.rank))

        if isinstance(m1, AltAlignment):
            setattr(model, n1, NewAltAlignment(m1.rank,m1.rank, mode,
                mom))
            #setattr(model, n1, NewAltAlignment(m1.rank, convert[m1.rank]                ))

def state_switch(model, state=0):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            state_switch(m1, state)
        if isinstance(m1, Alignment):
            m1.state = state

        if isinstance(m1, ResamplingDoubleFactConv2d):
            m1.state=state


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
            new_module = NewAlignment(m1.rank, m1.rank)
            new_module.alignment = m1.alignment.detach()
            new_module.state = 1
            setattr(model, n1, new_module)

def biason(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            biason(m1)
        if isinstance(m1, ResamplingDoubleFactConv2d):
            module=m1
            new_module = ResamplingDoubleFactConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True )
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['weight']
            new_sd['tri1_vec'] = old_sd['tri1_vec']
            new_sd['tri2_vec'] = old_sd['tri2_vec']
            new_module.load_state_dict(new_sd)
            new_module.tri1_vec.requires_grad=False
            new_module.tri2_vec.requires_grad=False
            torch.nn.init.zeros_(new_module.bias)
            new_module.bias.requires_grad=True
            setattr(model, n1, new_module)
        elif isinstance(m1, nn.Linear):
            for name, param in m1.named_parameters():
                param.requires_grad=False
        else:
            for name, param in m1.named_parameters():
                param.requires_grad = False


def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1,nn.Conv2d):# ResamplingDoubleFactConv2d):
            m1.resample()

def reset(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            reset(m1)
        if isinstance(m1, NewAlignment):
            m1.reset()


def load_model(args, model):
    #src="../saved-models/Long_Cifar_ResNets/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/retry_recent_new_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_rainbow_cifar/"
    #src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/recent_new_rainbow_cifar/"
    #src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/state_switch_rainbow_cifar/"
    src="/home/mila/m/muawiz.chaudhary/scratch/factconvs/saved_models/top3_recent_new_rainbow_cifar/"
    src="/home/mila/v/vivian.white/scratch/factconvs/saved_models/rainbow_cifar/"
    #run_name\
    #= "{}_batchsize_{}_rank_{}_resample_{}_width_{}_seed_{}_epochs_{}_k_{}_lr{}".format(args.net,
    run_name=\
    "{}_batchsize_{}_rank_{}_resample_{}_width_{}_seed_{}_epochs_{}_k_{}".format(args.net,
            args.batchsize, args.rank,
            #1 if args.width == 0.125 else args.double, args.resample,
            args.resample,
              args.width, args.seed, args.num_epochs,
              args.channel_k, args.lr)
    sd = torch.load(src+run_name+"/model.pt")
    #for key in sd.keys():
    #    if "resampling_weight" in key:
    #        temp = sd[key.replace("resampling_weight", "weight")]
    ##        #sd[key.replace("resampling_weight", "weight")] = sd[key]
    #        sd[key] = temp
    #print(sd.keys())
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
parser.add_argument('--resampling_seed', default=0, type=int, help='seed to use')
parser.add_argument('--double', default=0, type=int, help='seed to use')
parser.add_argument('--channel_k', default=512, type=int, help='seed to use')
parser.add_argument('--optimize', default=1, type=int, help='seed to use')
parser.add_argument('--statistics', default=1, type=int, help='seed to use')
parser.add_argument('--resample', default=0, type=int, help='seed to use')
parser.add_argument('--batchsize', default=256, type=int, help='seed to use')
parser.add_argument('--rank', default=200, type=int, help='seed to use')
parser.add_argument('--bias', default=0, type=int, help='seed to use')
parser.add_argument('--width', type=float, default=1, help='resnet width scale factor')
parser.add_argument('--gmm', default=0, type=int, help='seed to use')
parser.add_argument('--t', default=0, type=float, help='seed to use')
parser.add_argument('--bn_statistics', default=1, type=int)
parser.add_argument('--align_statistics', default=1, type=int)
parser.add_argument('--replace_align', default=0, type=int)
parser.add_argument('--mom', default=1.0, type=float, help='momentum value')
parser.add_argument('--mode', default='momentum', type=str, choices=['average', 'momentum'], help='average or momentum collection')

args = parser.parse_args()

# TRy$ THE PRINCIPLED WAy$
def resample_infinite(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample_infinite(m1)
        if isinstance(m1, ResamplingDoubleFactConv2d):
            sd = m1.state_dict()
            total = 1
            #print("HERE WE ARE")
            #for i in range(0, 10000):
            #    m1.resample()
            #    new_sd = m1.state_dict()
            #    sd['resampling_weight'] = sd['resampling_weight']  + new_sd['resampling_weight'] 
            #    total += 1
            t = args.t
            a = t
            b = math.sqrt(1-t**2)
            sd['resampling_weight'] = a*sd['resampling_weight'] + b*sd['weight']
            #print(torch.mean(sd['resampling_weight']))
            #print(torch.std(sd['resampling_weight']))
            #print(torch.var(sd['resampling_weight']))
            m1.load_state_dict(sd)


                #= sd['resampling_weight']*(total/(total+1))\
                #+ new_sd['resampling_weight'] * (1/(total+1))


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
#batch=args.batchsize#1000
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

run_name = "explore_corrected_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)
run_name = "no_bias_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)

run_name = "true_learn_bias_10_epochs_rainbow_width_{}_seed_{}".format(args.width, args.seed)
run_name = "bias_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)
run_name = "conv_width_{}_seed_{}".format(args.width, args.seed)

#run_name = "eigh_width_{}_optimize_{}_statistics_{}_seed_{}".format(args.width, args.optimize, args.statistics, args.seed)
#run_name = "conv_width_{}_seed_{}".format(args.width, args.seed)
#run_name = "conv_width_{}_seed_{}".format(args.width, args.seed)
run_name = "{}_width_{}_seed_{}".format(args.name, args.width, args.seed)

run_name= "optimize_{}_statistics_{}_resampleseed_{}_{}_batchsize_{}_rank_{}_{}_resample_{}_width_{}_seed_{}_epochs_{}".format(args.optimize,
        args.statistics, args.resampling_seed, args.net, args.batchsize, args.rank, args.double,
        args.resample, args.width, args.seed, args.num_epochs)

print("Args.net: ", args.net)
#print("Net: ", net)
set_seeds(0)
#set_seeds(args.seed)
if args.double:
    net = net.double()
net = net.to(device)
wandb_dir = "/home/mila/v/vivian.white/scratch/wandb/"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

run = wandb.init(project="FactConv", entity="whitev4", config=args,
        group="final_loading_probing_align_resnet_cifar", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)
sd = load_model(args, net)
net.load_state_dict(sd)
#print(net)

criterion = nn.CrossEntropyLoss()
parameters = net.linear.parameters() if "alt_aligned" not in args.net else net.resnet.linear.parameters()
optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9,
        weight_decay=5e-4)
#optimizer = optim.SGD(filter(lambda p: p.requires_grad,  net.parameters()), lr=0.1,
#optimizer = optim.Adam(filter(lambda p: p.requires_grad,  net.parameters()),
#        lr=0.0001,)
#
#optimizer = optim.Adam(filter(lambda p: p.requires_grad,  net.parameters()),
#        lr=0.0001,)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#        T_max=args.num_epochs)
args.resample=0
logger = {}

def replace_align(net):                                            
    for (n1, m1) in net.named_children():                      
        if len(list(m1.children())) > 0:                         
            replace_align(m1)                                          
        if isinstance(m1, Alignment): 
            new_module = NewAlignment(m1.rank, m1.rank, args.mode, args.mom)
            setattr(net, n1, new_module)

def reset_optimizer(net, opt):
    parameters = net.linear.parameters() if "alt_aligned" not in args.net else net.resnet.linear.parameters()
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9,weight_decay=5e-4)

# Training
def train(epoch, state=0, num_ensemble_samples=10):
    print('\nEpoch: %d' % epoch)

    if args.bn_statistics:
        def train_bn(net):                                            
            for (n1, m1) in net.named_children():                      
                if len(list(m1.children())) > 0:                         
                    train_bn(m1)                                          
                if isinstance(m1, nn.BatchNorm2d): 
                    m1.train()
        train_bn(net)
    if args.align_statistics:
        def train_align(net):                                            
            for (n1, m1) in net.named_children():                      
                if len(list(m1.children())) > 0:                         
                    train_align(m1)                                          
                if isinstance(m1, NewAlignment): 
                    m1.train()
        train_align(net)
    else:
        net.eval()
    #net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.double:
            inputs = inputs.double()

        optimizer.zero_grad()

        # 0 : only used for Conv or SRF networks 
        # 1 : generated, 2 : ensemble 3 : reference
        if state == 1:
            outputs = net(inputs)[:inputs.shape[0]]
        elif state == 0:
            outputs = net(inputs)
        elif state==2:
            for i in range(0, num_ensemble_samples):
                resample(net)
                if outputs is None:
                    outputs = net(inputs)[:inputs.shape[0]]/num_ensemble_samples
                else:
                    outputs += net(inputs)[:inputs.shape[0]]/num_ensemble_samples
        else: 
            outputs = net(inputs)[inputs.shape[0]:]
 
        loss = criterion(outputs, targets)
        loss.backward()

        if args.optimize:
            optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    logger["train_accuracy"] = 100.*correct/total


def test(epoch, state=1, num_ensemble_samples=10):
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

            # 0 : only used for Conv or SRF networks 
            # 1 : generated, 2 : ensemble 3 : reference
            outputs = None
            if state == 1:
                outputs = net(inputs)[:inputs.shape[0]]
            elif state==2:
                for i in range(0, num_ensemble_samples):
                    resample(net)
                    if outputs is None:
                        outputs = net(inputs)[:inputs.shape[0]]/num_ensemble_samples
                    else:
                        outputs += net(inputs)[:inputs.shape[0]]/num_ensemble_samples
            elif state == 0:
                outputs = net(inputs)
            else: 
                outputs = net(inputs)[inputs.shape[0]:]
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
recorder = {"sampled_net": 0.0, "reference_net": 0.0, "ensemble_10": 0.0,
        "adapted_1":0.0, "adapted_2":0.0, "adapted_3":0.0, "adapted_4":0.0,
        "adapted_5":0.0}
#set_seeds(args.seed)
set_seeds(args.resampling_seed)
net.cuda()
#state_switch(net, 0)
#biason(net)
#resample_infinite(net)
#factconv(net)
#print(net)

# wanna evaluate generated, reference, and ensemble + adapted networks.
if "align" in args.net:
    # one sample
    set_seeds(args.resampling_seed)
    resample(net)
    resampled_sd = net.state_dict()

    # Vivian Unadapted MiniBatch Alignment Experiment
    print("Unadapted MiniBatch")
    net.load_state_dict(resampled_sd)
    args.optimization = 0
    test(0,0)
    recorder['unadapted_minibatch'] = logger['accuracy']

    # Vivian Adapted MiniBatch Alignment Experiment With BatchNorm
    # Use the pretrained model
    print("Adapted MiniBatch")
    net.load_state_dict(resampled_sd)
    args.bn_statistics = 1
    args.optimization = 0
    reset_optimizer(net, optimizer)
    for epoch in range(0, 5):
        train(epoch, 0)
        test(epoch, 0)
        recorder['adapted_minibatch_{}'.format(epoch+1)] = logger['accuracy']

    # Vivian Unadapted TrainSet Alignment Experiment
    # no bn stats collection and/or linear layer adaptation
    print("Unadapted TrainSet")
    net.load_state_dict(resampled_sd)
    replace_align(net)
    args.bn_statistics = 0
    args.optimization = 0
    for epoch in range(0, 5):
        train(epoch, 0)
        test(epoch, 0)
        recorder['unadapted_trainset_{}'.format(epoch+1)] = logger['accuracy']

    reset(net)
    
    # Vivian Adapted TrainSet Alignment Experiment With BatchNorm
    print("Adapted TrainSet")
    net.load_state_dict(resampled_sd)
    args.bn_statistics = 1
    args.optimizer = 0
    reset_optimizer(net, optimizer)
    for epoch in range(0, 5):
        train(epoch, 0)
        test(epoch, 0)
        recorder['adapted_trainset_{}'.format(epoch+1)] = logger['accuracy']

    # reference network
#    test(0, 3)
#    recorder['reference_net'] = logger['accuracy']
    #
    #test(0, 2, 10)
    #recorder['ensemble_10'] = logger['accuracy']

    # Run the Momentum-based collecting
#    resample(net)
#    if args.statistics:
#        realign(net, mode='momentum', mom=1.0)
    #print(net)
#    for epoch in range(0, 5):
#        train(epoch, 1)
#        test(epoch, 1)
#        recorder['adapted_mom_{}'.format(epoch+1)] = logger['accuracy']

    # Reset the NewAltAlignment covariance to 0
#    sd = load_model(args, net)
#    net.load_state_dict(sd)
#    reset(net)

    # Run the Average-based collecting
#    if args.statistics:
#        realign(net, mode='average')
#    for epoch in range(0, 5):
#        train(epoch, 1)
#        test(epoch, 1)
#        recorder['adapted_avg_{}'.format(epoch+1)] = logger['accuracy']
else:
    # just evaluate standard conv or SRF networks
    test(0, 0)
    recorder['reference_net'] = logger['accuracy']
run.log(recorder)
