'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
import torchvision.transforms as transforms
import time 
import os
import argparse
import copy

from pytorch_cifar_utils import progress_bar, set_seeds
from test_models_safety import PostExp, PreExp
import wandb
from distutils.util import strtobool
from resnet import ResNet18
from vgg import VGG
import numpy as np
import gc
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.preferred_linalg_library('magma')
    
def save_model(args, model):
    #assert False
    src  = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/covar_new_testing_rainbow_models/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    
    torch.save(model.state_dict(), model_dir+ "/model.pt")
    torch.save(args, model_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--affine', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Batch Norm affine True or False')
parser.add_argument('--fact', type=lambda x: bool(strtobool(x)), 
                        default=True, help='FactNet True or False')
parser.add_argument('--width', default=0.125, type=float, help='width')
parser.add_argument('--sampling', type=str, default='random',
        choices=['rainbow', 'random'], help="which sampling to use")
args = parser.parse_args()
if args.width == 1.0:
    args.width = 1
if args.width == 2.0:
    args.width = 2
if args.width == 4.0:
    args.width = 4
if args.width == 8.0:
    args.width = 8


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
    trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

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
            #output = self._conv_forward(input, composite_weight, self.bias)
            new_sd['weight'] = composite_weight
            new_module.load_state_dict(new_sd)
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



net=ResNet18()
net.to(device)
replace_layers_agnostic(net, args.width)
if args.fact:
    replace_layers(net)
if not args.affine:
    replace_affines(net)

if args.fact and args.affine:
    if args.width == 8:
        sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/width_8/8scale_final/fact_model.pt")
    else:
        sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/fact_model.pt".format(args.width))
elif not args.fact and args.affine:
    if args.width == 8:
        sd=torch.load("/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/width_8/8scale_final/conv_model.pt")
    else:
        sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/conv_model.pt".format(args.width))
elif args.fact and not args.affine:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_2/{}scale_final/fact_model.pt".format(args.width))
elif not args.fact and not args.affine:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_2/{}scale_final/conv_model.pt".format(args.width))
net.load_state_dict(sd)
net.to(device)
net_new = copy.deepcopy(net)
net_new.to(device)
print(net_new)
replace_layers_fact(net)
net.to(device)

net.train()
net_new.train()

def calc_svd(A, name=''):
    gc.collect()
    torch.cuda.empty_cache()
    #print("svd")
    u, s, vh = torch.linalg.svd(
                   A, full_matrices=False,
               # driver="gesvd"
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
    #print("Condition Number {}:".format(name), s[0]/s[-1])
    alignment = u  @ vh  # (C_in_reference, C_in_generated)
    del u 
    del s
    del vh
    return alignment
 
def calc_svd_eigh(A, name=''):
    gc.collect()
    torch.cuda.empty_cache()
   # print("eigh")
    A_T_A = A.T@A 
    V_val, Vn = torch.linalg.eigh(A_T_A)
    del A_T_A
    V_val = V_val.flip(0)
    Vn    = Vn.fliplr().T
    Sn = (1e-6 + V_val.abs()).sqrt()
    Sn_inv = (1/Sn).diag()
    Un = A @ Vn.T @ Sn_inv
    alignment  = Un @ Vn
    #print("Condition Number {}:".format(name), Sn[0]/Sn[-1])
    del V_val
    del Vn
    del Sn
    del Sn_inv
    del Un
    #alignment = Un  @ Vn  # (C_in_reference, C_in_generated)
    return alignment
 

@torch.no_grad()
def recurse_layers(model, new_model):
    for (n1, m1), (n2, m2) in zip(model.named_children(), new_model.named_children()):
        if len(list(m1.children())) > 0:
            recurse_layers(m1, m2)
        if isinstance(m1, nn.Conv2d):# and  m1.in_channels == 4096:
            print("conv")
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

            #new_module = weight_Alignment(m1, m2, new_module)

            old_sd = m1.state_dict()
            new_sd = new_module.state_dict()
            current_sd = m2.state_dict()
            og_copy = new_sd['weight']

            if m1.bias is not None:
                new_sd['bias'] = old_sd['bias']
            if "tri1_vec" in current_sd.keys():
                new_sd['tri1_vec']=current_sd['tri1_vec']
                new_sd['tri2_vec']=current_sd['tri2_vec']
            copy_weight = current_sd['weight']
            copy_weight_gen = og_copy
            copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
            copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
            weight_cov = (copy_weight_gen@copy_weight)	
            alignment = calc_svd_eigh(weight_cov, name="Weight")
            new_weight= og_copy
            new_weight = new_weight.reshape(new_weight.shape[0], -1)
            new_weight = new_weight@alignment # C_in_reference to C_in_generated
            new_weight = new_weight.reshape(old_sd['weight'].shape)
            new_module.register_buffer("weight_align", alignment)
            new_sd['weight_align'] = alignment
            new_sd['weight'] = new_weight
            #
            ##this is the spot where
            ## we can do weight alignment
            ##   for fact net, this means aligning with the random noise
            ##   for conv net, this could mean aligning with a. W OR b. U 
            ## we can do colored-covariance specification
            ##   for fact net, this means just using it's R matrix
            ##   for conv net, this could mean doing nothing (if aligning with W), or use S and V if we did b. 
 
            ##multiply = weight_cov.T@weight_cov

            #gc.collect()
            #torch.cuda.empty_cache()
            #u, s, vh = torch.linalg.svd(
            #       weight_cov, full_matrices=False,
            #   # driver="gesvd"
            #)  # (C_in_reference, R), (R,), (R, C_in_generated)
            #alignment = u  @ vh  # (C_in_reference, C_in_generated)
 
            #s = time.time()
            #V_val, Vn = torch.linalg.eigh(multiply)
            #del multiply
            #print(time.time()-s)
            #V_val = V_val.flip(0)
            #Vn    = Vn.fliplr().T
            #Sn = (1e-6 + V_val.abs()).sqrt()
            #Sn_inv = (1/Sn).diag()
            #Un = weight_cov @ Vn.T @ Sn_inv
            #s = time.time()
            #alignment  = Un @ Vn
            #print(torch.norm(Un), torch.norm(Vn))

            #del weight_cov
            #del s
            #del vh
            #del u
            #del Sn_inv
            #del Un

            #del V_val
            #del Vn
            #del Un
            #del Sn_inv
            #gc.collect()
            #torch.cuda.empty_cache()
            if m1.in_channels != 3:# and False:
                activation = []
                other_activation = []

                def define_hook(m):
                    def store_hook(mod, inputs, outputs):
                        #inputs[0] = b x c x w x h
                        #def
                        # retry with their code exactly. 
                        # # remove weight alignment
                        #inputs[0].permute(0,2,3,1).reshape(-1, inputs[0].shape[1]))
                        x = inputs[0]
                        x = x.permute(0, 2, 3, 1)
                        x = x.reshape((-1, x.shape[-1]))
                        activation.append(x)
                        raise Exception("Done")
                    return store_hook

                print(m1)
                print(m2)

                hook_handle_1 = m1.register_forward_hook(define_hook(m1))
                hook_handle_2 = m2.register_forward_hook(define_hook(m2))

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
                        covar = activation[-2].T @ activation[-1]
                        assert (covar.isfinite().all())
                    else: 
                        covar += activation[-2].T @ activation[-1]
                        assert (covar.isfinite().all())
                    activation = []
                    other_activation = []
                covar /= total
                hook_handle_1.remove()
                hook_handle_2.remove()
                print("done with covariance_calc")
                #print("logdet")
                #print(torch.logdet(covar))
                #print(torch.logdet(covar))
                #print("logdet covariacne")
                #print(torch.logdet(covar@covar.T))
                #print(torch.logdet(covar.T@covar))
 
                #print("cond num")
                #cond = torch.linalg.svdvals(covar)#, driver="gesvda")
                ##U,S,V = torch.linalg.svd(A, full_matrices=False)
                #b = cond[0]/cond[-1]
                #print(b)
 
                #u, s, v = torch.linalg.svd(covar, full_matrices=False,)
                #driver="gesvd")
                #b = s[0]/s[-1]
                #print(b)
                #b = s[-1]/s[0]
                #print(b)
                #align = u @ v
                align = calc_svd(covar, name="Cross-Covariance")
                new_module.register_buffer("input_align", align)
                new_sd['input_align'] = align
                hook_handle_pre_forward = new_module.register_forward_pre_hook(return_hook())
                #del u
                #del s
                #del v
                del covar
                del activation
                del other_activation
            new_module.load_state_dict(new_sd)
            #if isinstance(new_module, FactConv2dPreExp):
            #    ## simple module
            #    print("TESTING FACT REPLACEMENT")
            #    fact_module = nn.Conv2d(
            #            in_channels=new_module.in_channels,
            #            out_channels=new_module.out_channels,
            #            kernel_size=new_module.kernel_size,
            #            stride=new_module.stride, padding=new_module.padding, 
            #            bias=True if new_module.bias is not None else False)
            #    old_sd = new_module.state_dict()
            #    new_sd = fact_module.state_dict()
            #    if new_module.bias is not None:
            #        new_sd['bias'] = old_sd['bias']
            #    U1 = new_module._tri_vec_to_mat(new_module.tri1_vec, new_module.in_channels //
            #        new_module.groups, new_module.scat_idx1)
            #    U2 = new_module._tri_vec_to_mat(new_module.tri2_vec,
            #            new_module.kernel_size[0] * new_module.kernel_size[1],
            #            new_module.scat_idx2)
            #    U = torch.kron(U1, U2) 
            #    matrix_shape = (new_module.out_channels, new_module.in_features)
            #    composite_weight = torch.reshape(
            #        torch.reshape(new_module.weight, matrix_shape) @ U,
            #        new_module.weight.shape
            #    )
            #    new_sd['weight'] = composite_weight
            #    if 'weight_align' in old_sd.keys():
            #        new_sd['weight_align'] = old_sd['weight_align']
            #        shape  = fact_module.in_channels*fact_module.kernel_size[0]*fact_module.kernel_size[1]
            #        fact_module.register_buffer("weight_align",torch.zeros((shape, shape)))
            #    if 'input_align' in old_sd.keys():
            #        new_sd['input_align'] = old_sd['input_align']
            #        out_shape = fact_module.in_channels
            #        fact_module.register_buffer("input_align",torch.zeros((out_shape, out_shape)))
            #    if new_module.in_channels != 3:
            #        hook_handle_pre_forward.remove()
            #        hook_handle_pre_forward = fact_module.register_forward_pre_hook(return_hook())
            #    fact_module.load_state_dict(new_sd)
            #    fact_module.to(device)
            #    new_module = fact_module
            #    print("FACT REPLACEMENT:", new_module)
            #    del U1 
            #    del U2
            #    del U

            setattr(new_model, n1, new_module)

        if isinstance(m1, nn.Linear):# and False:
            print("linear")
            new_module = nn.Linear(m1.in_features, m1.out_features, bias=True
                    if m1.bias is not None else False).to(device)
            old_sd = m1.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['weight']
            if m1.bias is not None:
                new_sd['bias'] = old_sd['bias']
            activation = []
            other_activation = []

            hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
                    activation.append(inputs[0]))

            hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
                    other_activation.append(inputs[0]))
            covar = None
            total = 0
            print("starting covariance_calc")
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
            #print("done with covariance_calc")
            #print("logdet")
            #print(torch.logdet(covar))
            #print(torch.logdet(covar))
            #print("logdet covariacne")
            #print(torch.logdet(covar@covar.T))
            #print(torch.logdet(covar.T@covar))
 
            #print("cond num")
            #cond = torch.linalg.svdvals(covar)#, driver="gesvda")
            ##U,S,V = torch.linalg.svd(A, full_matrices=False)
            #b = cond[0]/cond[-1]
            #print(b)
 
            hook_handle_1.remove()
            hook_handle_2.remove()
            #u, s, v = torch.linalg.svd(covar, full_matrices=False)#, driver="gesvd")
            #b = s[0]/s[-1]
            #print(b)
            #b = s[-1]/s[0]
            #print(b)
 
            #align = u @ v
            align = calc_svd_eigh(covar, name="Cross-Covariance")
            new_weight = new_sd['weight']
            new_weight = torch.moveaxis(new_weight, source=1,
                    destination=-1)
            new_weight = new_weight@align
            new_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                    destination=1)        
            new_module.load_state_dict(new_sd)
            setattr(new_model, n1, new_module)
        if isinstance(m1,nn.BatchNorm2d):
            print("BatchieNormie")
            m1.train()
            m2.train()
            m1.reset_running_stats()
            m2.reset_running_stats()
            handel_1 = m1.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
            handel_2 = m2.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
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
            handel_1.remove()
            handel_2.remove()
            m1.eval()
            m2.eval()

def weight_Alignment(m1, m2, new_module):
    # reference model state dict
    print("we go here")
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
    reference_weight = ref_sd['weight']
    generated_weight = new_gaussian
    
    #reshape to outdim x indim*spatial
    reference_weight = reference_weight.reshape(reference_weight.shape[0], -1)
    generated_weight = generated_weight.reshape(generated_weight.shape[0], -1)
    #compute transpose, giving indim*spatial x outdim
    #generated_weight = generated_weight.T
    
    #compute weight cross-covariance indim*spatial x indim*spatial
    #TODO REFACTOR TO HAVE REF FIRST. OUTDIM x OUTDIM 
    weight_cov = (generated_weight.T@reference_weight)
    #weight_cov = (reference_weight@generated_weight.T)
    alignment = calc_svd_eigh(weight_cov, name="Weight alignment")
    
    # outdim x indim x spatial
    final_gen_weight = new_gaussian
    # outdim x indim*spatial
    final_gen_weight = final_gen_weight.reshape(final_gen_weight.shape[0], -1)
    # outdim x indim*spatial
    final_gen_weight = final_gen_weight@alignment
    #final_gen_weight = alignment@final_gen_weight
    # outdim x indim x spatial
    loading_sd['weight'] = final_gen_weight.reshape(ref_sd['weight'].shape)
    loading_sd['weight_align'] = alignment
    new_module.register_buffer("weight_align", alignment)
    new_module.load_state_dict(loading_sd)
    return new_module
 



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
            print("old sd:",torch.norm(old_sd['weight']))
            print("new sd:",torch.norm(new_sd['weight']))
            #new_sd['weight'] *= torch.norm(old_sd['weight'])/torch.norm(new_sd['weight'])
            print("new sd:",torch.norm(new_sd['weight']))
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
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
            #new_sd['weight'] *= torch.norm(old_sd['weight'])/torch.norm(new_sd['weight'])
            new_module.load_state_dict(new_sd)
            print("new sd:", torch.norm(new_sd['weight']))
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

set_seeds(1)
#net=VGG("VGG11", args.bn_on)
criterion = nn.CrossEntropyLoss()
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
    #run.log({"accuracy":acc})

print("testing Res{}Net18 with width of {}".format("Fact" if args.fact else "Conv", args.width))
pretrained_acc, og_loss = test(0, net)

set_seeds(1)
s=time.time()
if args.sampling == "rainbow":
    rainbow_sampling(net_new)
    net_new.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_new(inputs)
else: 
    net_new.train()
    recurse_layers(net, net_new)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_new(inputs)

run_name= "{}_Res{}Net18_Width_{}_{}_affine".format(args.sampling.capitalize(), "Fact"
        if args.fact else "Conv", str(args.width), "No" if not args.affine else
        "Yes")


print("TOTAL TIME:", time.time()-s)
turn_off_grads(net_new)
##
##
## 
#
optimizer = optim.SGD(filter(lambda param: param.requires_grad, net_new.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
#print("testing rainbow sampling")
print("testing {} sampling at width {}".format(args.sampling, args.width))
net_new.eval()

run_name= "correctly_saved_network_{}_Res{}Net18_Width_{}_{}_affine".format(args.sampling.capitalize(), "Fact"
        if args.fact else "Conv", str(args.width), "No" if not args.affine else
        "Yes")

args.name = run_name
print(net_new)
sampled_acc, sampled_loss = test(0, net_new)
#assert False
#print("training rainbow sampling classifier head for 10 epochs")
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
run = wandb.init(project="random_project", config=args,
        group="recomputing_conv_rainbow_sampling", name=run_name, dir=wandb_dir)
run.log(logger)


args.name += "_trained_classifier_head"
save_model(args, net_new)
#wandb.watch(net, log='all', log_freq=1)
