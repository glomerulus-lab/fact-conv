import torch
import torch.nn as nn
from conv_modules import FactConv2d
from learnable_cov import V1_init

def replace_layers_factconv2d(model):
    '''
    Replace nn.Conv2d layers with FactConv2d
    '''
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_factconv2d(module)
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2d(
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

def replace_layers_scale(model, scale=1):
    '''
    Replace nn.Conv2d layers with a different scale
    '''
    prev_out_ch = 0
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_scale(module,scale)
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
            new_module = nn.Linear(int(module.in_features * scale), 10)
            setattr(model, n, new_module)

def turn_off_grad(model, covariance):
    '''
    Turn off gradients in tri1_vec or tri2_vec to turn off
    channel or spatial covariance learning
    '''
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            turn_off_grad(module, covariance)
        if isinstance(module, FactConv2d):
            for name, param in module.named_parameters():
                if covariance == "channel":
                    if "tri1_vec" in name:
                        param.requires_grad = False
                if covariance == "spatial":
                    if "tri2_vec" in name:
                        param.requires_grad = False

def init_V1_layers(model, bias):
    '''
    Initialize every FactConv2d layer with V1-inspired
    spatial weight init
    '''
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            init_V1_layers(module, bias)
        if isinstance(module, FactConv2d):
            kernel_size = 3 
            center = ((kernel_size - 1) / 2, (kernel_size - 1) / 2)
            V1_init(module, size=2, spatial_freq=0.1, scale=1, center=center)
            for name, param in module.named_parameters():
                if "weight" in name:
                    param.requires_grad = False

                if bias:
                    if "bias" in name:
                            param.requires_grad = False

