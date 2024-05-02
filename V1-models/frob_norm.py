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

from resnet import ResNet18
from vgg import VGG

from test_models_safety import PostExp, PreExp
import copy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', type=str, default='pretrained_fact', choices=['pretrained_fact', 'aca_no_wa_fact', 'aca_wa_input_fact', 'wa_input_fact'], help="which convmodule to use")
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), 
                        default=False, help='bias=True or False')

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
            
def return_hook():
        def hook(mod, inputs):
            shape = inputs[0].shape
            inputs_permute = inputs[0].permute(1,0,2,3).reshape(inputs[0].shape[1], -1)
            reshape = (mod.input_align@inputs_permute).reshape(shape[1],
                    shape[0], shape[2],
                    shape[3]).permute(1, 0, 2, 3)
            return reshape 
        return hook
    #hook_handle_pre_forward = new_module.register_forward_pre_hook(return_hook())

def add_align(model, weight_align=True, input_align=True):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            add_align(module, weight_align, input_align)
        if isinstance(module, nn.Conv2d):
            #print(module)
            shape = module.in_channels*module.kernel_size[0]*module.kernel_size[1]
            out_shape = module.in_channels
            if weight_align:
                module.register_buffer("weight_align",torch.zeros((shape, shape)))
            if out_shape != 3 and input_align:
                module.register_buffer("input_align",torch.zeros((out_shape, out_shape)))
            #old_sd = module.state_dict()
            #old_sd['weight_align']=None
            #old_sd['input_align']=None
            #module.load_state_dict(old_sd, False)
            #setattr(model, n, module)
            #old_sd = module.state_dict()

 
def add_hook(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            add_hook(module)
        if isinstance(module, nn.Conv2d) and module.in_channels != 3:
            hook_handle_pre_forward = module.register_forward_pre_hook(return_hook())
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

def replace_layers_fact(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_fact(module)
        if isinstance(module, FactConv2dPreExp):
            ## simple module
            new_module = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            #new_sd['weight'] = old_sd['weight']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            U1 = module._tri_vec_to_mat(module.tri1_vec, module.in_channels //
                module.groups,module.scat_idx1)
            U2 = module._tri_vec_to_mat(module.tri2_vec, module.kernel_size[0] * module.kernel_size[1],
                    module.scat_idx2)
            U = torch.kron(U1, U2) 
            matrix_shape = (module.out_channels, module.in_features)
            composite_weight = torch.reshape(
                torch.reshape(module.weight, matrix_shape) @ U,
                module.weight.shape
            )
            if "input_align" in old_sd.keys():
                new_weight = torch.moveaxis(composite_weight, source=1,
                        destination=-1)
                new_weight = new_weight@old_sd['input_align']
                composite_weight = torch.moveaxis(new_weight, source=-1,
                        destination=1)        
 
            new_sd['weight'] = composite_weight
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)



def test(net, epoch=0):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))


set_seeds(0)
print("Loading FactNet")
fact_net = ResNet18()
replace_layers(fact_net)
sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/1scale_final/fact_model.pt")
fact_net.load_state_dict(sd)
fact_net.to(device)
test(fact_net)

replace_layers_fact(fact_net)
fact_net.to(device)
test(fact_net)


print("Loading ACAFacttNet")
aca_net = ResNet18()
replace_layers(aca_net)
add_align(aca_net, False)
sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/final_refactor_covar_new_testing_rainbow_models/final_Ours_ResFactNet18_Width_1_Yes_affine_ACA_On_WA_Off_InputWA_False/model.pt")
aca_net.load_state_dict(sd)
add_hook(aca_net)
aca_net.to(device)
test(aca_net)

replace_layers_fact(aca_net)
aca_net.to(device)
test(aca_net)



print("Loading ACAIWAFacttNet")
aca_iwa_net = ResNet18()
replace_layers(aca_iwa_net)
add_align(aca_iwa_net, True)
sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/final_refactor_covar_new_testing_rainbow_models/final_Ours_ResFactNet18_Width_1_Yes_affine_ACA_On_WA_On_InputWA_True/model.pt")
aca_iwa_net.load_state_dict(sd)
add_hook(aca_iwa_net)
aca_iwa_net.to(device)
test(aca_iwa_net)

replace_layers_fact(aca_iwa_net)
aca_iwa_net.to(device)
test(aca_iwa_net)


print("Loading IWAFacttNet")
iwa_net = ResNet18()
replace_layers(iwa_net)
add_align(iwa_net, True, False)
sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/final_refactor_covar_new_testing_rainbow_models/final_Ours_ResFactNet18_Width_1_Yes_affine_ACA_Off_WA_On_InputWA_True/model.pt")
iwa_net.load_state_dict(sd)
iwa_net.to(device)
test(iwa_net)

replace_layers_fact(iwa_net)
iwa_net.to(device)
test(iwa_net)


set_seeds(0)
set_seeds(0)

def frob_norm(ref_weight, gen_weight):
    top = torch.pow(torch.linalg.norm(ref_weight-gen_weight),2).item()
    bottom = torch.pow(torch.linalg.norm(ref_weight), 2).item()
    return top/bottom

layer_results = []
@torch.no_grad()
def get_norm(model, new_model, name=""):
    for (n1, m1), (n2, m2) in zip(model.named_children(), new_model.named_children()):
        if len(list(m1.children())) > 0:
            get_norm(m1, m2, name)
        if isinstance(m1, nn.Conv2d):
            ref_weight = m1.weight
            gen_weight = m2.weight
            result = frob_norm(ref_weight, gen_weight)
            layer_results.append(result)

        if isinstance(m1, nn.Linear):
            ref_weight = m1.weight
            gen_weight = m2.weight
            result = frob_norm(ref_weight, gen_weight)
            layer_results.append(result)

get_norm(fact_net, fact_net, name="fact vs fact")
fact_vs_fact = copy.deepcopy(layer_results)
layer_results = []
print()
get_norm(fact_net, aca_iwa_net, name="fact vs ACA + IWA")
fact_vs_aca_iwa = copy.deepcopy(layer_results)
layer_results = []
print()
get_norm(fact_net, aca_net, name="fact vs ACA")
fact_vs_aca = copy.deepcopy(layer_results)
layer_results = []
print()
get_norm(fact_net, iwa_net, name="fact vs IWA")
fact_vs_iwa = copy.deepcopy(layer_results)
layer_results = []
print()

wandb_dir = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
run_name = args.net

run = wandb.init(project="random_project", config=args,
        #group="pytorch_cifar_better_tracked_og", name=run_name, dir=wandb_dir)
        group="Fact_Frob_Norm", name=run_name, dir=wandb_dir)
logger = {}
for i in range(0, len(fact_vs_fact)):
    logger["fact_vs_fact_layer_{}".format(i)] = fact_vs_fact[i],
    logger["fact_vs_aca_layer_{}".format(i)] = fact_vs_aca[i]
    logger["fact_vs_iwa_layer_{}".format(i)] = fact_vs_iwa[i]
    logger["fact_vs_aca_iwa_layer_{}".format(i)] = fact_vs_aca_iwa[i]
run.log(logger)
