import torch
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, convnext_tiny
from torchvision.models.resnet import resnet18
from ConvModules import FactConv2dPreExp
from linear_modules import FactLinear

def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module)
        # replace conv2d layers with factconvs
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    groups=module.groups, dilation=module.dilation,
                    padding_mode=module.padding_mode,
                    bias=True if module.bias is not None else False)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['weight']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)
        if isinstance(module, nn.Linear):
            if module.out_features == 1000:
                print("Classifier head")
            else:
                new_module = FactLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=True if module.bias is not None else False)
                old_sd = module.state_dict()
                new_sd = new_module.state_dict()
                new_sd['weight'] = old_sd['weight']
                if module.bias is not None:
                    new_sd['bias'] = old_sd['bias']
                new_module.load_state_dict(new_sd)
                setattr(model, n, new_module)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#save_dir = "/home/mila/v/vivian.white/scratch/v1-models/saved-models/imagenet/"
os.makedirs(save_dir, exist_ok=True)
test = torch.randn(10, 3, 224, 224).to(device)

conv_model = alexnet()
conv_model = conv_model.to(device)
#torch.save(conv_model.state_dict(), '{}/conv_alexnet_init.pt'.format(save_dir))
fact_model = copy.deepcopy(conv_model)
fact_model.to(device)
replace_layers(fact_model)
fact_model(test)

conv_model = convnext_tiny()
conv_model = conv_model.to(device)
#torch.save(conv_model.state_dict(), '{}/conv_convnext_tiny_init.pt'.format(save_dir))
fact_model = copy.deepcopy(conv_model)
fact_model.to(device)
replace_layers(fact_model)
fact_model(test)

conv_model = resnet18()
conv_model = conv_model.to(device)
#torch.save(conv_model.state_dict(), '{}/conv_resnet_init.pt'.format(save_dir))
fact_model = copy.deepcopy(conv_model)
fact_model.to(device)
replace_layers(fact_model)
fact_model(test)

