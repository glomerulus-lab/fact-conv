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
import wandb
from distutils.util import strtobool
from resnet import ResNet18
from vgg import VGG
from test_models_safety import PostExp, PreExp
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

    
def save_model(args, model):
    src = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_rainbow_models/"
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
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--net', type=str, default='vgg', choices=['vgg', 'vggbn',
    'resnet', 'factnetv1', 'factnetdefault', 'vggfact', 'vggbnfact'], help="which convmodule to use")
parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                    help="initialization for spatial filters for LearnableCov models")
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--bn_on', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Batch Norm True or False')
parser.add_argument('--fact', type=lambda x: bool(strtobool(x)), 
                        default=True, help='FactNet True or False')
parser.add_argument('--sampling', type=str, default='rainbow',
        choices=['rainbow', 'random'], help="which sampling to use")
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
from ConvModules import FactConv2dPreExp
def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module)
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False)
            setattr(model, n, new_module)


@torch.no_grad()
def rainbow_sampling(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            rainbow_sampling(module)
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
        if isinstance(module, nn.Conv2d):
            ## simple module
            if isinstance(module, FactConv2dPreExp):
                new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)
            else:
                new_module = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            print(torch.mean(torch.pow(old_sd['weight']-new_sd['weight'], 2)))
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
            #new_sd['weight'] = old_sd['weight']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            copy_weight = old_sd['weight']
            copy_weight_gen = new_sd['weight']
            copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
            copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
            weight_cov = (copy_weight_gen@copy_weight)##.T
            u, s, vh = torch.linalg.svd(
                   weight_cov, full_matrices=False
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
            alignment = u  @ vh  # (C_in_reference, C_in_generated)
            new_weight= new_sd['weight'] #generated_model[j].weight
            new_weight = new_weight.reshape(new_weight.shape[0], -1)
            new_weight = new_weight@alignment # C_in_reference to C_in_generated
            new_weight = new_weight.reshape(module.weight.shape)
          # Set the new weights in the generated model.
          # NOTE: this an intermediate model, as sampling the j-th layer means that the j+1-th layer is no longer aligned.
          # As such, if evaluated as is, its accuracy would be that of a random guess.
            new_sd['weight'] = new_weight
            new_module.load_state_dict(new_sd)
            print(torch.mean(torch.pow(old_sd['weight']-new_sd['weight'], 2)))
            print()
            setattr(model, n, new_module)



@torch.no_grad()
def random_sampling(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            random_sampling(module)
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
        if isinstance(module, nn.Conv2d):
            ## simple module
            if isinstance(module, FactConv2dPreExp):
                new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)
            else:
                new_module = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)

            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            print(torch.mean(torch.pow(old_sd['weight']-new_sd['weight'], 2)))
            print()
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)

def turn_off_grads(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            turn_off_grads(module)
        else:
            if isinstance(module, nn.Linear) and module.out_features == 10:
                grad=True
            else:
                grad=False
            for param in module.parameters():
                param.requires_grad = grad

set_seeds(0)
net=VGG("VGG11", args.bn_on)
if args.fact:
    replace_layers(net)
if args.bn_on and args.fact:
    sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_pytorch_tta/vggbnfactfinal/trial_1/model.pt")
elif args.bn_on and not args.fact:
    sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_pytorch_tta/vggbn_redofinal/trial_1/model.pt")
elif not args.bn_on and args.fact:
    sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_pytorch_tta/vggfactfinal/trial_1/model.pt")
elif not args.bn_on and not args.fact:
    sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_pytorch_tta/vgg_redofinal/trial_1/model.pt")
net.load_state_dict(sd)
net.to(device)

criterion = nn.CrossEntropyLoss()
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
    print("accuracy:", acc)
    #run.log({"accuracy":acc})

print("testing VGG_{}Net with BN {}".format("Fact" if args.fact else "Conv",
    "On" if args.bn_on else "Off"))
test(net)

set_seeds(0)
if args.sampling == "rainbow":
    rainbow_sampling(net)
else: 
    random_sampling(net)
turn_off_grads(net)
net.train()


for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
 

optimizer = optim.SGD(filter(lambda param: param.requires_grad, net.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
#print("testing rainbow sampling")
print("testing {} sampling".format(args.sampling))
net.eval()
test(0)
#print("training rainbow sampling classifier head for 10 epochs")
print("training classifier head of {} sampled mode for {} epochs".format(args.sampling, args.epochs))
for i in range(0, args.epochs):
    net.train()
    train(i)
    net.eval()
    test(i)

