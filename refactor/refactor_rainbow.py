'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import wandb
import numpy as np

import time 
import os
import argparse
import copy
import gc
from distutils.util import strtobool

from pytorch_cifar_utils import progress_bar, set_seeds
from models.resnet import ResNet18
from conv_modules import FactConv2d

def save_model(args, model):
    src = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/refactoring/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    
    torch.save(model.state_dict(), model_dir+ "/model.pt")
    torch.save(args, model_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--aca', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Activation Cross-Covariance Alignment')
parser.add_argument('--wa', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Weight alignment True=Yes False=No')
parser.add_argument('--in_wa', type=lambda x: bool(strtobool(x)), 
                        default=True, help='input=True output=False')
parser.add_argument('--fact', type=lambda x: bool(strtobool(x)), 
                        default=True, help='FactNet True or False')
parser.add_argument('--width', default=0.125, type=float, help='width')
parser.add_argument('--sampling', type=str, default='ours',
        choices=['ours', 'theirs'], help="which sampling to use")

args = parser.parse_args()

if args.width == 1.0:
    args.width = 1
if args.width == 2.0:
    args.width = 2
if args.width == 4.0:
    args.width = 4
if args.width == 8.0:
    args.width = 8

print("Sampling: {} Width: {} Fact: {} ACA: {} WA: {} In_WA: {}".format(args.sampling,
    args.width, args.fact, args.aca, args.wa, args.in_wa))

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
    trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')


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


def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module)
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2d(
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
        if isinstance(module, FactConv2d):
            ## simple module
            new_module = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
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
            new_sd['weight'] = composite_weight
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)


net=ResNet18()
net.to(device)
replace_layers_agnostic(net, args.width)
if args.fact:
    replace_layers(net)
if args.fact:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/fact_model.pt".format(args.width))
elif not args.fact:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/conv_model.pt".format(args.width))
net.load_state_dict(sd)
net.to(device)
net_new = copy.deepcopy(net)
net_new.to(device)
print(net_new)
replace_layers_fact(net)
net.to(device)

net.train()
net_new.train()

set_seeds(args.seed)
criterion = nn.CrossEntropyLoss()


#traditional way of calculating svd. can be a bit unstable sometimes tho
def calc_svd(A, name=''):
    u, s, vh = torch.linalg.svd(
                   A, full_matrices=False,
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
    alignment = u  @ vh  # (C_in_reference, C_in_generated)
    return alignment
 

#i've been finding this way of calculating svd to be more stable. 
def calc_svd_eigh(A, name=''):
    A_T_A = A.T@A 
    V_val, Vn = torch.linalg.eigh(A_T_A)
    V_val = V_val.flip(0)
    Vn    = Vn.fliplr().T
    Sn = (1e-6 + V_val.abs()).sqrt()
    Sn_inv = (1/Sn).diag()
    Un = A @ Vn.T @ Sn_inv
    alignment  = Un @ Vn
    return alignment
 

#used in activation cross-covariance calculation
#input align hook
def return_hook():
    def hook(mod, inputs):
        shape = inputs[0].shape
        inputs_permute = inputs[0].permute(1,0,2,3).reshape(inputs[0].shape[1], -1)
        reshape = (mod.input_align@inputs_permute).reshape(shape[1],
                shape[0], shape[2],
                shape[3]).permute(1, 0, 2, 3)
        return reshape 
    return hook


#settings
# our, wa true, aca true (fact, conv)
# our wa false, aca true (fact, conv)
# our wa false aca false (fact, conv) [Just Random]
# our was true aca false (fact, conv)
#
# theirs wa true, aca true (conv)
# theirs wa false, aca true (conv)
# theirs wa false aca false (conv) [Just Random]
# theirs was true aca false (conv)
#
# this function does not do an explicit specification of the colored covariance
@torch.no_grad()
def our_rainbow_sampling(model, new_model):
    for (n1, m1), (n2, m2) in zip(model.named_children(), new_model.named_children()):
        if len(list(m1.children())) > 0:
            our_rainbow_sampling(m1, m2)
        if isinstance(m1, nn.Conv2d):
            if isinstance(m2, FactConv2d):
                new_module = FactConv2d(
                    in_channels=m2.in_channels,
                    out_channels=m2.out_channels,
                    kernel_size=m2.kernel_size,
                    stride=m2.stride, padding=m2.padding, 
                    bias=True if m2.bias is not None else False)
            else:
                new_module = nn.Conv2d(
                    in_channels=int(m2.in_channels),
                    out_channels=int(m2.out_channels),
                    kernel_size=m2.kernel_size,
                    stride=m2.stride, padding=m2.padding, 
                    groups = m2.groups,
                    bias=True if m2.bias is not None else False).to(device)

            if args.sampling == 'ours' and args.wa:
                # right now this function does not do an explicit specification of the colored covariance
                new_module = weight_Alignment(m1, m2, new_module, in_dim=args.in_wa)
            if args.sampling == 'theirs':
                # for conv only
                if args.wa:
                    new_module = weight_Alignment_With_CC(m1, m2, new_module)
                else:
                    new_module = colored_Covariance_Specification(m1, m2, new_module)
            # this step calculates the activation cross-covariance alignment (ACA)
            if m1.in_channels != 3 and args.aca:
                new_module = conv_ACA(m1, m2, new_module)
            # converts fact conv to conv. this is for sake of speed.
            #if isinstance(new_module, FactConv2d):
            #    new_module = fact_2_conv(new_module)
           #changes the network module
            setattr(new_model, n1, new_module)

        #only computes the ACA
        if isinstance(m1, nn.Linear) and args.aca:
            new_module = linear_ACA(m1, m2, new_model)
            setattr(new_model, n1, new_module)
        #just run stats through
        if isinstance(m1, nn.BatchNorm2d):
            batchNorm_stats_recalc(m1, m2)


def weight_Alignment(m1, m2, new_module, in_dim=True):
    # reference model state dict
    ref_sd = m1.state_dict()
    # generated model state dict - uses reference model weights. for now
    gen_sd = m2.state_dict()
    
    # module with random init - to be loaded to model
    loading_sd = new_module.state_dict()
    new_gaussian = loading_sd['weight']
    
    # carry over old bias. only matters when we work with no batchnorm networks
    if m1.bias is not None:
        loading_sd['bias'] = ref_sd['bias']
    # carry over old colored covariance. only matters with fact-convs
    if "tri1_vec" in gen_sd.keys():
        loading_sd['tri1_vec']=gen_sd['tri1_vec']
        loading_sd['tri2_vec']=gen_sd['tri2_vec']

    #this is the spot where
    # we can do weight alignment
    #   for fact net, this means aligning with the random noise
    #   for conv net, this could mean aligning with a. W OR b. U 
    # we can do colored-covariance specification
    #   for fact net, this means just using it's R matrix
    #   for conv net, this could mean doing nothing (if aligning with W), or use S and V if we did b. 
    # in this function, we just align with W and don't specify the mulit-color covariance 
    
    # IF FACT: we align the generated factnet with the reference fact net's noise
    # IF CONV: we align the generated convnet with the reference conv net's weight matrix
    reference_weight = gen_sd['weight']
    generated_weight = new_gaussian
    
    #reshape to outdim x indim*spatial
    reference_weight = reference_weight.reshape(reference_weight.shape[0], -1)
    generated_weight = generated_weight.reshape(generated_weight.shape[0], -1)
    #compute transpose, giving indim*spatial x outdim
    
    #compute weight cross-covariance indim*spatial x indim*spatial
    #TODO REFACTOR TO HAVE REF FIRST. OUTDIM x OUTDIM 
    if in_dim:
        print("Input Weight Alignment")
        weight_cov = (generated_weight.T@reference_weight)
        alignment = calc_svd(weight_cov, name="Weight alignment")
        
        # outdim x indim x spatial
        final_gen_weight = new_gaussian
        # outdim x indim*spatial
        final_gen_weight = final_gen_weight.reshape(final_gen_weight.shape[0], -1)
        # outdim x indim*spatial
        final_gen_weight = final_gen_weight@alignment
    else:
        print("Output Weight Alignment")
        weight_cov = (reference_weight@generated_weight.T)
        alignment = calc_svd(weight_cov, name="Weight alignment")
        
        # outdim x indim x spatial
        final_gen_weight = new_gaussian
        # outdim x indim*spatial
        final_gen_weight = final_gen_weight.reshape(final_gen_weight.shape[0], -1)
        # outdim x indim*spatial
        final_gen_weight = alignment@final_gen_weight

    loading_sd['weight'] = final_gen_weight.reshape(ref_sd['weight'].shape)
    loading_sd['weight_align'] = alignment
    new_module.register_buffer("weight_align", alignment)
    new_module.load_state_dict(loading_sd)
    return new_module
 

def conv_ACA(m1, m2, new_module):
    print("Convolutional Input Activations Alignment")
    activation = []
    other_activation = []
    # this hook grabs the input activations of the conv layer
    # rearanges the vector so that the width by height dim is 
    # additional samples to the covariance
    # bwh x c
    def define_hook(m):
        def store_hook(mod, inputs, outputs):
            #from bonner lab tutorial
            x = inputs[0]
            x = x.permute(0, 2, 3, 1)
            x = x.reshape((-1, x.shape[-1]))
            activation.append(x)
            raise Exception("Done")
        return store_hook
    
    hook_handle_1 = m1.register_forward_hook(define_hook(m1))
    hook_handle_2 = m2.register_forward_hook(define_hook(m2))
    
    print("Starting Sample Cross-Covariance Calculation")
    covar = None
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        try:
            outputs1 = net(inputs)
        except Exception:
            pass
        try:
            outputs2 = net_new(inputs)
        except Exception:
            pass
        total+= inputs.shape[0]
        if covar is None:
            #activation is bwh x c
            covar = activation[0].T @ activation[1]
            assert (covar.isfinite().all())
        else: 
            #activation is bwh x c
            covar += activation[0].T @ activation[1]
            assert (covar.isfinite().all())
        activation = []
        other_activation = []

    #c x c
    covar /= total
    hook_handle_1.remove()
    hook_handle_2.remove()
    print("Sample Cross-Covariance Calculation finished")
    align = calc_svd(covar, name="Cross-Covariance")
    new_module.register_buffer("input_align", align)

    # this hook takes the input to the conv, aligns, then returns
    # to the conv the aligned inputs
    hook_handle_pre_forward  = new_module.register_forward_pre_hook(return_hook())
    return new_module


def linear_ACA(m1, m2, new_model):
    print("Linear Input Activations Alignment")
    new_module = nn.Linear(m1.in_features, m1.out_features, bias=True
            if m1.bias is not None else False).to(device)
    ref_sd = m1.state_dict()
    loading_sd = new_module.state_dict()
    loading_sd['weight'] = ref_sd['weight']
    if m1.bias is not None:
        loading_sd['bias'] = ref_sd['bias']
    activation = []
    other_activation = []
    
    hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
            activation.append(inputs[0]))
    
    hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
            other_activation.append(inputs[0]))
    covar = None
    total = 0
    print("Starting Sample Cross-Covariance Calculation")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs1 = net(inputs)
        outputs2 = net_new(inputs)
        total+= inputs.shape[0]
        if covar is None:
            covar = activation[0].T @ other_activation[0]
        else: 
            covar += activation[0].T @ other_activation[0]
        activation = []
        other_activation = []
    covar /= total
    
    hook_handle_1.remove()
    hook_handle_2.remove()
    
    print("Sample Cross-Covariance Calculation finished")
    align = calc_svd(covar, name="Cross-Covariance")
    new_weight = loading_sd['weight']
    new_weight = torch.moveaxis(new_weight, source=1,
            destination=-1)
    new_weight = new_weight@align
    loading_sd['weight'] = torch.moveaxis(new_weight, source=-1,
            destination=1)        
    new_module.load_state_dict(loading_sd)
    return new_module


def batchNorm_stats_recalc(m1, m2):
    print("Calculating Batch Statistics")
    m1.train()
    m2.train()
    m1.reset_running_stats()
    m2.reset_running_stats()
    handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
    handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
    m1.to(device)
    m2.to(device)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        try:
            outputs1 = net(inputs)
        except Exception:
            pass
        try:
            outputs2 = net_new(inputs)
        except Exception:
            pass
    handle_1.remove()
    handle_2.remove()
    m1.eval()
    m2.eval()
    print("Batch Statistics Calculation Finished")


def weight_Alignment_With_CC(m1, m2, new_module, Un=None, Sn=None, Vn=None):
    print("NOT SUPPOSED TO BE HERE")
    # reference model state dict
    ref_sd = m1.state_dict()
    # generated model state dict - uses reference model weights. for now
    gen_sd = m2.state_dict()

    # module with random init - to be loaded to model
    loading_sd = new_module.state_dict()
    new_gaussian = loading_sd['weight']
    
    # carry over old bias. only matters when we work with no batchnorm networks
    if m1.bias is not None:
        loading_sd['bias'] = ref_sd['bias']
    # carry over old colored covariance. only matters with fact-convs
    if "tri1_vec" in gen_sd.keys():
        loading_sd['tri1_vec']=gen_sd['tri1_vec']
        loading_sd['tri2_vec']=gen_sd['tri2_vec']

    old_weight = ref_sd['weight']
    A = old_weight.reshape(old_weight.shape[0], -1)
    A_T_A = A.T@A 
    V_val, Vn = torch.linalg.eigh(A_T_A)
    del A_T_A
    V_val = V_val.flip(0)
    Vn    = Vn.fliplr().T
    Sn = (1e-6 + V_val.abs()).sqrt()
    Sn_inv = (1/Sn).diag()
    Un = A @ Vn.T @ Sn_inv 
    white_gaussian = torch.randn_like(Un)

    copy_weight = Un 
    copy_weight_gen = white_gaussian
    copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
    copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
    weight_cov = (copy_weight_gen@copy_weight)

    alignment = calc_svd(weight_cov, name="Weight")
    new_weight = white_gaussian  
    new_weight = new_weight.reshape(new_weight.shape[0], -1)
    new_weight = new_weight@alignment # C_in_reference to C_in_generated

    new_module.register_buffer("weight_align", alignment)
    loading_sd['weight_align'] = alignment
    colored_gaussian = white_gaussian @ (Sn[:,None]* Vn)
    loading_sd['weight'] = colored_gaussian.reshape(old_weight.shape)
    new_module.load_state_dict(loading_sd)
    return new_module


# this function does not do an explicit specification of the colored covariance
@torch.no_grad()
def colored_Covariance_Specification(m1, m2, new_module, Un=None, Sn=None, Vn=None):
    print("NOT HERE")
    # reference model state dict
    ref_sd = m1.state_dict()
    # generated model state dict - uses reference model weights. for now
    gen_sd = m2.state_dict()

    # module with random init - to be loaded to model
    loading_sd = new_module.state_dict()
    new_gaussian = loading_sd['weight']
    
    # carry over old bias. only matters when we work with no batchnorm networks
    if m1.bias is not None:
        loading_sd['bias'] = ref_sd['bias']
    # carry over old colored covariance. only matters with fact-convs
    if "tri1_vec" in gen_sd.keys():
        loading_sd['tri1_vec']=gen_sd['tri1_vec']
        loading_sd['tri2_vec']=gen_sd['tri2_vec']

    old_weight = ref_sd['weight']
    A = old_weight.reshape(old_weight.shape[0], -1)
    A_T_A = A.T@A 
    V_val, Vn = torch.linalg.eigh(A_T_A)
    del A_T_A
    V_val = V_val.flip(0)
    Vn    = Vn.fliplr().T
    Sn = (1e-6 + V_val.abs()).sqrt()
    Sn_inv = (1/Sn).diag()
    Un = A @ Vn.T @ Sn_inv 
    white_gaussian = torch.randn_like(Un)

    colored_gaussian = white_gaussian @ (Sn[:,None]* Vn)
    loading_sd['weight'] = colored_gaussian.reshape(old_weight.shape)
    new_module.load_state_dict(loading_sd)
    return new_module


def fact_2_conv(new_module):
    ## simple module
    print("Replacing FactConv")
    fact_module = nn.Conv2d(
            in_channels=new_module.in_channels,
            out_channels=new_module.out_channels,
            kernel_size=new_module.kernel_size,
            stride=new_module.stride, padding=new_module.padding, 
            bias=True if new_module.bias is not None else False)

    old_sd = new_module.state_dict()
    new_sd = fact_module.state_dict()

    if new_module.bias is not None:
        new_sd['bias'] = old_sd['bias']

    U1 = new_module._tri_vec_to_mat(new_module.tri1_vec, new_module.in_channels //
        new_module.groups, new_module.scat_idx1)
    U2 = new_module._tri_vec_to_mat(new_module.tri2_vec,
            new_module.kernel_size[0] * new_module.kernel_size[1],
            new_module.scat_idx2)
    U = torch.kron(U1, U2) 

    matrix_shape = (new_module.out_channels, new_module.in_features)
    composite_weight = torch.reshape(
        torch.reshape(new_module.weight, matrix_shape) @ U,
        new_module.weight.shape
    )

    new_sd['weight'] = composite_weight
    if 'weight_align' in old_sd.keys():
        new_sd['weight_align'] = old_sd['weight_align']
        shape  = fact_module.in_channels*fact_module.kernel_size[0]*fact_module.kernel_size[1]
        fact_module.register_buffer("weight_align",torch.zeros((shape, shape)))
    if 'input_align' in old_sd.keys():
        new_sd['input_align'] = old_sd['input_align']
        out_shape = fact_module.in_channels
        fact_module.register_buffer("input_align",torch.zeros((out_shape, out_shape)))
        if new_module.in_channels != 3:
            #fact check this
            for key in list(new_module._forward_pre_hooks.keys()):
                del new_module._forward_pre_hooks[key]
            hook_handle_pre_forward  = fact_module.register_forward_pre_hook(return_hook())
    fact_module.load_state_dict(new_sd)
    fact_module.to(device)
    new_module = fact_module
    print("FACT REPLACEMENT:", new_module)
    return new_module
 
           
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


def train(epoch, net):
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


def test(epoch, net):
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
    return acc, test_loss


print("testing Res{}Net18 with width of {}".format("Fact" if args.fact else "Conv", args.width))
pretrained_acc, og_loss = test(0, net)

set_seeds(args.seed)
s=time.time()
our_rainbow_sampling(net, net_new)
net_new.train()
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net_new(inputs)
print("TOTAL TIME:", time.time()-s)
turn_off_grads(net_new)
optimizer = optim.SGD(filter(lambda param: param.requires_grad, net_new.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
print("testing {} sampling at width {}".format(args.sampling, args.width))
net_new.eval()

run_name = "refactor"
args.name = run_name
print(net_new)

sampled_acc, sampled_loss = test(0, net_new)
save_model(args, net_new)
accs = []
test_losses= []
print("training classifier head of {} sampled model for {} epochs".format(args.sampling, args.epochs))
for i in range(0, args.epochs):
    net_new.train()
    train(i, net_new)
    net_new.eval()
    acc, loss_test =test(i, net_new)
    test_losses.append(loss_test)
    accs.append(acc)

logger ={"pretrained_acc": pretrained_acc, "sampled_acc": sampled_acc,
        "first_epoch_acc":accs[0], "third_epoch_acc": accs[2],
        "tenth_epoch_acc":accs[args.epochs-1], 'width':args.width,
        "og_loss":og_loss, "sampled_loss":sampled_loss,
        "first_epoch_loss":test_losses[0], "third_epoch_loss": test_losses[2],
        "tenth_epoch_loss":test_losses[args.epochs-1], 'width':args.width}

wandb_dir = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
group_name = "refactor"
run = wandb.init(project="random_project", config=args,
        group=group_string, name=run_name, dir=wandb_dir)
run.log(logger)

args.name += "_trained_classifier_head"
save_model(args, net_new)
