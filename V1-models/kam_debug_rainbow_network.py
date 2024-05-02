'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
import torchvision.transforms as transforms

import copy
import os
import argparse

from pytorch_cifar_utils import progress_bar, set_seeds
from test_models_safety import PostExp, PreExp
import wandb
from distutils.util import strtobool
from resnet import ResNet18
from layers_model import Sequential_ThreeLayer_CIFAR10
from vgg import VGG
from test_models_safety import PostExp, PreExp
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True

import pdb

    
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
parser.add_argument('--net', type=str,
                    default='RSN', choices=['RSN', 'ResNet'], help="architecture")
                    #default='vgg', choices=['vgg', 'vggbn', 'resnet', 'factnetv1', 'factnetdefault', 'vggfact', 'vggbnfact'], help="which convmodule to use")
parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                    help="initialization for spatial filters for LearnableCov models")
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--affine', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Batch Norm affine True or False')
parser.add_argument('--fact', type=lambda x: bool(strtobool(x)), 
                        default=True, help='FactNet True or False')
parser.add_argument('--width', default=1, type=float, help='width')
parser.add_argument('--sampling', type=str, default='ours',
        choices=['ours', 'theirs'], help="which sampling to use")
parser.add_argument('--aca', type=lambda x: bool(strtobool(x)),
                                            default=True, help='activation alignment True/False')
parser.add_argument('--wa', type=lambda x: bool(strtobool(x)),
                                            default=True, help='weight alignment True/False')
parser.add_argument('--wa_dim', type=str, default="input", choices=['input', 'output'],
                    help='weight alignment dimension, "input" or "output"')
args = parser.parse_args()
if args.width == 1.0:
    args.width = 1
if args.width == 2.0:
    args.width = 2

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
    root='./data', train=True, download=True, transform=transform_train)#transform_train)
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

def replace_layers_agnostic(model, scale=1):
    '''
    Replaces layers with themselves (?!)
    '''
    # TODO: explain what is going on here
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

def replace_affines(model):
    '''
    Set BatchNorm2d layers to have 'affine=False'
    '''
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_affines(module)
        if isinstance(module, nn.BatchNorm2d):
            ## simple module
            new_module = nn.BatchNorm2d(
                    num_features=module.num_features,
                    eps=module.eps, momentum=module.momentum,
                    affine=False,
                    track_running_stats=module.track_running_stats)
            setattr(model, n, new_module)

def replace_layers_conv(model):
    '''
    Replace Conv2d layers with FactConv2dPreExp
    '''
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_conv(module)
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
    '''
    Replace FactConv2dPreExp layers with Conv2d
    Computes the weight matrix corresponding to using weights from Fact model
    '''
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
            #output = self._conv_forward(input, composite_weight, self.bias)
            new_sd['weight'] = composite_weight
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)

#traditional way of calculating svd. can be a bit unstable sometimes tho
def calc_alignment(cross_cov, name=''):
    u, s, vh = torch.linalg.svd(
                   cross_cov, full_matrices=False,
               # driver="gesvd"
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
    alignment = u  @ vh  # (C_in_reference, C_in_generated)
    return alignment
 
#i've been finding this way of calculating svd to be more stable. 
def calc_alignment_eigh(cross_cov, name=''):
    A_T_A = cross_cov.T @ cross_cov
    V_val, Vn = torch.linalg.eigh(A_T_A)
    V_val = V_val.flip(0)
    Vn    = Vn.fliplr().T
    Sn = (1e-6 + V_val.abs()).sqrt()
    Sn_inv = (1/Sn).diag()
    Un = cross_cov @ Vn.T @ Sn_inv
    alignment  = Un @ Vn
    return alignment
 
#used in activation cross-covariance calculation
#input align pre_hook
def alignment_hook():
    def hook(mod, inputs):
        x = inputs[0] # b x c x w x h
        A = mod.input_align # c x c
        x = torch.moveaxis(x, source=1, destination=-1)
        x = x @ A.T
        return torch.moveaxis(x, source=-1, destination=1)
        # # b x c x w x h
        # shape = inputs[0].shape
        # # -> c x b x w x h
        # x = inputs[0].permute(1,0,2,3)
        # # -> c x (b * w * h)
        # x = x.reshape(inputs[0].shape[1], -1)
        # # align using stored matrix
        # x = mod.input_align @ x
        # # -> c x b x w x h -> b x c x w x h
        # x = x.reshape(shape[1], shape[0], shape[2], shape[3]).permute(1, 0, 2, 3)
        # return x
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
            # recurse if not a base module
            our_rainbow_sampling(m1, m2)

        if isinstance(m1, nn.Conv2d):
            if isinstance(m2, FactConv2dPreExp):
                new_module = FactConv2dPreExp(
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
            print(type(m2))
            print(f"weight: {new_module.weight.shape}")
            # if args.sampling == 'ours' and args.wa:
            #     # right now this function does not do an explicit specification of the colored covariance
            #     #print("HI")
            if args.wa: 
                new_module = weight_alignment(m1, m2, new_module, dim=args.wa_dim)
            # if args.sampling == 'theirs':
            #     # for conv only
            #     if args.wa:
            #         new_module = weight_alignment_With_CC(m1, m2, new_module)
            #     else:
            #         new_module = colored_Covariance_Specification(m1, m2, new_module)
            # this step calculates the activation cross-covariance alignment (ACA)
            if m1.in_channels != 3 and args.aca:
                new_module = conv_activation_alignment(m1, m2, new_module)
            # converts fact conv to conv. this is for sake of speed.
            #if isinstance(new_module, FactConv2dPreExp):
            #    new_module = fact_2_conv(new_module)
            #changes the network module
            setattr(new_model, n2, new_module)
        if isinstance(m1, nn.Linear) and args.aca:
            # only input activation alignment
            new_module = linear_activation_alignment(m1, m2, new_model)
            setattr(new_model, n2, new_module)
        if isinstance(m1, nn.BatchNorm2d):
            #just run stats through
            batchNorm_stats_recalc(m1, m2)

def weight_alignment(m1, m2, new_module, dim='input'):
    '''
    Align weights either by input (default) or output dim
    '''
    # reference model state dict
    print(f"weight alignment, {dim} dimension")
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
    #generated_weight = generated_weight.T

    if dim == 'input':
        #compute new-old cross-covariance (indim*spatial x indim*spatial)
        weight_cov = generated_weight.T @ reference_weight
        alignment = calc_alignment(weight_cov, name="Weight alignment")
        # outdim x indim x spatial
        final_gen_weight = new_gaussian
        # outdim x indim*spatial
        final_gen_weight = final_gen_weight.reshape(final_gen_weight.shape[0], -1)
        # (outdim x indim*spatial) @ (indim*spatial)
        final_gen_weight = final_gen_weight @ alignment
    elif dim == 'output':
        #compute old-new cross-covariance (outdim x outdim)
        weight_cov = reference_weight @ generated_weight.T
        alignment = calc_alignment(weight_cov, name="Weight alignment")
        # outdim x indim x spatial
        final_gen_weight = new_gaussian
        # outdim x indim*spatial
        final_gen_weight = final_gen_weight.reshape(final_gen_weight.shape[0], -1)
        # (outdim x outdim) @ (outdim x indim*spatial)
        final_gen_weight = alignment @ final_gen_weight

    # Reshape to orig dimensions
    # outdim x indim x spatial
    loading_sd['weight'] = final_gen_weight.reshape(ref_sd['weight'].shape)

    # Commenting out this augmentation to state_dict
    # loading_sd['weight_align'] = alignment
    # new_module.register_buffer("weight_align", alignment)
    new_module.load_state_dict(loading_sd)
    return new_module
 


def conv_activation_alignment(m1, m2, new_module):
    activation = []
    other_activation = []
    print("feature alignment: conv")
    
    # this hook grabs the activations of the conv layer
    # rearanges the vector so that the width by height dim is 
    # additional samples to the covariance
    # (b * w * h) x c
    def preactivation_hook(m):
        def store_hook(mod, inputs, outputs):
            # (from bonner lab tutorial)
            # b x c x w x h
            x = inputs[0]
            # put out channels to last
            x = x.permute(0, 2, 3, 1)
            # (b * w * h) x c
            x = x.reshape((-1, x.shape[-1]))
            activation.append(x)
            # keeps things from passing through later layers
            raise Exception("Done")
        return store_hook
    
    hook_handle_1 = m1.register_forward_hook(preactivation_hook(m1))
    hook_handle_2 = m2.register_forward_hook(preactivation_hook(m2))
    
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
            # old x new
            covar = activation[0].T @ activation[1]
            assert (covar.isfinite().all())
        else: 
            #activation is bwh x c
            covar += activation[0].T @ activation[1]
            assert (covar.isfinite().all())
        activation = []
        other_activation = []
    covar /= total
    hook_handle_1.remove()
    hook_handle_2.remove()

    # align: old <- new
    align = calc_alignment(covar, name="Cross-Covariance")
    new_module.register_buffer("input_align", align)
    # this hook takes the input to the conv, aligns, then returns
    # to the conv the aligned inputs
    hook_handle_pre_forward  = new_module.register_forward_pre_hook(alignment_hook())
    return new_module

def linear_activation_alignment(m1, m2, new_model):
    print("feature alignment: linear")
    new_module = nn.Linear(m1.in_features, m1.out_features, bias=True
            if m1.bias is not None else False).to(device)
    ref_sd = m1.state_dict()
    loading_sd = new_module.state_dict()
    loading_sd['weight'] = ref_sd['weight']
    if m1.bias is not None:
        loading_sd['bias'] = ref_sd['bias']
    activation = []
    new_activation = []

    # These hooks store the pre-activations
    hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
                                             activation.append(inputs[0]))
    hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
                                             new_activation.append(inputs[0]))
    covar = None
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs1 = net(inputs)
        outputs2 = net_new(inputs)
        total+= inputs.shape[0]
        if covar is None:
            covar = activation[0].T @ new_activation[0]
        else: 
            covar += activation[0].T @ new_activation[0]
        activation = []
        new_activation = []
    covar /= total
    
    hook_handle_1.remove()
    hook_handle_2.remove()

    # align: old <- new
    align = calc_alignment(covar, name="Cross-Covariance")
    # copy old weights
    new_weight = loading_sd['weight']
    # move channel to last axis
    new_weight = torch.moveaxis(new_weight, source=1,
                                destination=-1)
    # align acts on pre-activations, absorb into readout
    new_weight = new_weight @ align
    loading_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                                          destination=1)    
    new_module.load_state_dict(loading_sd)
    return new_module

def batchNorm_stats_recalc(m1, m2):
    """
    Pass one epoch through to allow the batchnorm to readjust
    """
    print("BatchieNormie")
    #m1.train()
    #m2.train()
    m1.reset_running_stats()
    m2.reset_running_stats()
    handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
    handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
    #handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs: pass)
    #handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs: pass)
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
    #m1.eval()
    #m2.eval()

def weight_alignment_With_CC(m1, m2, new_module, Un=None, Sn=None, Vn=None):
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
    alignment = calc_alignment(weight_cov, name="Weight")
    new_weight = white_gaussian  
    new_weight = new_weight.reshape(new_weight.shape[0], -1)
    new_weight = new_weight@alignment # C_in_reference to C_in_generated

    new_module.register_buffer("weight_align", alignment)
    loading_sd['weight_align'] = alignment
    colored_gaussian = white_gaussian @ (Sn[:,None]* Vn)#(Sn[:,None]* Vn)
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
    colored_gaussian = white_gaussian @ (Sn[:,None]* Vn)#(Sn[:,None]* Vn)
    loading_sd['weight'] = colored_gaussian.reshape(old_weight.shape)
    new_module.load_state_dict(loading_sd)
    return new_module
 
def fact_2_conv(new_module):
    ## simple module
    print("TESTING FACT REPLACEMENT")
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
            hook_handle_pre_forward  = fact_module.register_forward_pre_hook(alignment_hook())
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
                # Readout layer
                for param in module.parameters():
                    param.requires_grad = True              
            elif isinstance(module, nn.BatchNorm2d):
                # Don't mess with BatchNorm params
                pass
            else:
                for param in module.parameters():
                    param.requires_grad = False

                
# Initialize model
set_seeds(0)

print(f"Running {args.net}")
if args.net == 'ResNet':
    # ResNet setup:
    net=ResNet18()
    # Load in FactConv2d layers
    replace_layers_agnostic(net, args.width)

    if args.fact:
        replace_layers_conv(net)
    if not args.affine:
        replace_affines(net)

    # Now load in the saved model params
    if args.fact and args.affine:
        sd=torch.load("./models/for_kam/fact_model.pt")
    elif not args.fact and args.affine:
        sd=torch.load("./models/for_kam/conv_model.pt")
    elif args.fact and not args.affine:
        sd=torch.load("./models/for_kam/fact_model_no_affines.pt")
    elif not args.fact and not args.affine:
        sd=torch.load("./models/for_kam/conv_model_no_affines.pt")

    # #if args.fact and args.affine:
    # #    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_scale/{}scale_final/fact_model.pt".format(args.width))
    # #elif not args.fact and args.affine:
    # #    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_scale/{}scale_final/conv_model.pt".format(args.width))
    # #elif args.fact and not args.affine:
    # #    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/no_affines/{}scale_final/fact_model.pt".format(args.width))
    # #elif not args.fact and not args.affine:
    # #    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/no_affines/{}scale_final/conv_model.pt".format(args.width))

    # Put those params into the network
    net.load_state_dict(sd)
    net.to(device)
elif args.net == 'RSN':
    net=Sequential_ThreeLayer_CIFAR10(100,False)
    if args.fact:
        replace_layers_conv(net)
        sd = torch.load('models/RSN/fact_model.pt')
        net.load_state_dict(sd)
        net.to(device)

# Setup train/test functions
criterion = nn.CrossEntropyLoss()
def train(net):
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

def test(net):
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
    return acc
    #run.log({"accuracy":acc})

# Now actually do something with the network
# First evaluate test accuracy of saved params
print(f"testing {args.net}, Fact-{args.fact} with width of {args.width}")
pretrained_acc = test(net)

net_new = copy.deepcopy(net)
net_new.to(device)

# Now resample the weight params
set_seeds(0)
#if args.sampling == "rainbow":
net_new.train()
our_rainbow_sampling(net, net_new)
#else: 
#    random_sampling(net)

# Disable gradients of all but Linear layers
turn_off_grads(net_new)
# net_new.train() # just sets the mode
# # Feed batches through network to fix batchnorms
# for i in range(2):
#     print(f'batchnorm adaptation pass {i}')
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = net_new(inputs)
 

#print("testing rainbow sampling")
print("testing {} sampling at width {}".format(args.sampling, args.width))
net_new.eval()
sampled_acc = test(net_new)
print(f'accuracy after sampling {sampled_acc}')
optimizer = optim.SGD(filter(lambda param: param.requires_grad, net_new.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

accs = []
print("training classifier head of {} sampled model for {} epochs".format(args.sampling, args.epochs))
for i in range(0, args.epochs):
    print('\nEpoch: %d' % i)
    net_new.train()
    train(net_new)
    accs.append(test(net_new))
#logger ={"pretrained_acc": pretrained_acc, "sampled_acc": sampled_acc,
#        "first_epoch_acc":accs[0], "third_epoch_acc": accs[2],
#        "tenth_epoch_acc":accs[args.epochs-1], 'width':args.width}
#
#wandb_dir = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/wandb"
#os.makedirs(wandb_dir, exist_ok=True)
#os.chdir(wandb_dir)
#run_name = "{}_Res{}Net18_Width_{}".format(args.sampling.capitalize(), "Fact" if args.fact else "Conv", str(args.width))
#
#run = wandb.init(project="random_project", config=args,
#        group="rainbow_sampling", name=run_name, dir=wandb_dir)
#run.log(logger)
#wandb.watch(net, log='all', log_freq=1)


