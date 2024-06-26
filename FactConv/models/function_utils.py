import torch
import torch.nn as nn
from conv_modules import FactConv2d
from V1_covariance import V1_init


def recurse_preorder(model, callback):
    r = callback(model)
    if r is not model and r is not None:
        return r
    for n, module in model.named_children():
        r = recurse_preorder(module, callback)
        if r is not module and r is not None:
            setattr(model, n, r)
    return model


def replace_layers_factconv2d(model):
    '''
    Replace nn.Conv2d layers with FactConv2d
    '''
    def _replace_layers_factconv2d(module):
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
            return new_module
    return recurse_preorder(model, _replace_layers_factconv2d)


def replace_affines(model):
    '''
    Set BatchNorm2d layers to have 'affine=False'
    '''
    def _replace_affines(module):
        if isinstance(module, nn.BatchNorm2d):
            ## simple module
            new_module = nn.BatchNorm2d(
                    num_features=module.num_features,
                    eps=module.eps, momentum=module.momentum,
                    affine=False,
                    track_running_stats=module.track_running_stats)
            return new_module
    return recurse_preorder(model, _replace_affines)


def replace_layers_scale(model, scale=1):
    '''
    Replace nn.Conv2d layers with a different scale
    '''
    def _replace_layers_scale(module):
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
            return new_module
        if isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(int(module.num_features*scale),
                    affine=module.affine)
            return new_module
        if isinstance(module, nn.Linear):
            new_module = nn.Linear(int(module.in_features * scale), 10)
            return new_module
    return recurse_preorder(model, _replace_layers_scale)


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


def replace_layers_fact_with_conv(model):
    '''
    Replace FactConv2d layers with nn.Conv2d
    '''
    def _replace_layers_fact_with_conv(module):
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
            if 'weight_align' in old_sd.keys():
                new_sd['weight_align'] = old_sd['weight_align']
                shape  = new_module.in_channels*new_module.kernel_size[0]*new_module.kernel_size[1]
                new_module.register_buffer("weight_align",torch.zeros((shape, shape)))
            if 'input_align' in old_sd.keys():
                new_sd['input_align'] = old_sd['input_align']
                out_shape = new_module.in_channels
                new_module.register_buffer("input_align",torch.zeros((out_shape, out_shape)))
                if module.in_channels != 3:
                    #fact check this
                    for key in list(module._forward_pre_hooks.keys()):
                        del module._forward_pre_hooks[key]
                    hook_handle_pre_forward  = new_module.register_forward_pre_hook(return_hook())
            new_module.load_state_dict(new_sd)
            new_module.to(old_sd['weight'].device)
            return new_module
    return recurse_preorder(model, _replace_layers_fact_with_conv)


def turn_off_covar_grad(model, covariance):
    '''
    Turn off gradients in tri1_vec or tri2_vec to turn off
    channel or spatial covariance learning
    '''
    def _turn_off_covar_grad(module):
        if isinstance(module, FactConv2d):
            for name, param in module.named_parameters():
                if covariance == "channel":
                    if "tri1_vec" in name:
                        param.requires_grad = False
                if covariance == "spatial":
                    if "tri2_vec" in name:
                        param.requires_grad = False
    return recurse_preorder(model, _turn_off_covar_grad)
    
           
def turn_off_backbone_grad(model):
    '''
    Turn off gradients in backbone. For tuning just classifier layer
    '''
    def _turn_off_backbone_grad(module):
        if isinstance(module, nn.Linear) and module.out_features == 10:
            grad=True
        else:
            grad=False
        for param in module.parameters():
            param.requires_grad = grad
    return recurse_preorder(model, _turn_off_backbone_grad)


def init_V1_layers(model, bias):
    '''
    Initialize every FactConv2d layer with V1-inspired
    spatial weight init
    '''
    def _init_V1_layers(module):
        if isinstance(module, FactConv2d):
            center = ((module.kernel_size[0] - 1) / 2, (module.kernel_size[1] - 1) / 2)
            V1_init(module, size=2, spatial_freq=0.1, scale=1, center=center)
            for name, param in module.named_parameters():
                if "weight" in name:
                    param.requires_grad = False
                if bias:
                    if "bias" in name:
                            param.requires_grad = False
    return recurse_preorder(model, _init_V1_layers)

