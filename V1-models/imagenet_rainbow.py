import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, resnet50, convnext_base,\
convnext_base, convnext_tiny, alexnet, mobilenet_v3_small
from ConvModules import FactConv2dPreExp

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from imagenet_utils import validate, train
import os
import copy 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#net = #convnext_tiny(weights="IMAGENET1K_V1").cuda()
net = resnet18(weights="IMAGENET1K_V1").cuda()
net_new = copy.deepcopy(net)
net_new.to(device)
criterion = nn.CrossEntropyLoss().to(device)
traindir = os.path.join("/network/datasets/imagenet.var/imagenet_torchvision/", 'train')
valdir = os.path.join("/network/datasets/imagenet.var/imagenet_torchvision/", 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
print(net)

train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
scale = 2
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1024*scale, shuffle=True,
    num_workers=4, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1024*scale, shuffle=False,
    num_workers=4, pin_memory=False)
print("validating")
#validate(val_loader, net, criterion)


@torch.no_grad()
def recurse_layers(model, new_model):
    for (n1, m1), (n2, m2) in zip(model.named_children(), new_model.named_children()):
        if len(list(m1.children())) > 0:
            recurse_layers(m1, m2)
        if isinstance(m1, nn.Conv2d):
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


            old_sd = m1.state_dict()
            new_sd = new_module.state_dict()
            og_copy = new_sd['weight']

            if m1.bias is not None:
                new_sd['bias'] = old_sd['bias']
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
 

            #old_weight = old_sd['weight']
            #u, s, v = torch.linalg.svd(old_weight.reshape(old_weight.shape[0], -1), full_matrices=False)
            #white_gaussian = torch.randn_like(u)
            #colored_gaussian = white_gaussian @ (s[..., None] * v)
            #new_sd['weight'] = colored_gaussian.reshape(old_weight.shape)

            copy_weight = old_sd['weight']
            copy_weight_gen = og_copy #new_sd['weight']
            copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
            copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
            weight_cov = (copy_weight_gen@copy_weight)##.T
            u, s, vh = torch.linalg.svd(
                   weight_cov, full_matrices=False
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
            alignment = u  @ vh  # (C_in_reference, C_in_generated)
            new_weight= og_copy#new_sd['weight'] #generated_model[j].weight
            new_weight = new_weight.reshape(new_weight.shape[0], -1)
            new_weight = new_weight@alignment # C_in_reference to C_in_generated
            new_weight = new_weight.reshape(old_sd['weight'].shape)
            new_module.register_buffer("weight_align", alignment)
            new_sd['weight_align'] = alignment
            # Set the new weights in the generated model.
            # NOTE: this an intermediate model, as sampling the j-th layer means that the j+1-th layer is no longer aligned.
            # As such, if evaluated as is, its accuracy would be that of a random guess.
            new_sd['weight'] = new_weight

            if m1.in_channels != 3:
                activation = []
                #other_activation = []

                def define_hook():
                    def store_hook(mod, inputs, outputs):
                        activation.append(inputs[0].permute(0,2,3,1).reshape(-1,
                            inputs[0].shape[1]))
                        #act.append(inputs[0].permute(0,2,3,1).reshape(-1,
                        #    inputs[0].shape[1]))
                        raise Exception("Done")
                    return store_hook

                #hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
                #        activation.append(inputs[0].permute(0,2,3,1).reshape(-1,
                #            inputs[0].shape[1])))
                #hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
                #        other_activation.append(inputs[0].permute(0,2,3,1).reshape(-1,
                #            inputs[0].shape[1])))

                #hook_handle_1 = m1.register_forward_hook(define_hook(activation))
                #hook_handle_2 = m2.register_forward_hook(define_hook(other_activation))
                hook_handle_1 = m1.register_forward_hook(define_hook())
                hook_handle_2 = m2.register_forward_hook(define_hook())

                covar = None
                total = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
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
                        #covar = activation[0].T @ other_activation[0]
                    else: 
                        covar = activation[-2].T @ activation[-1]
                        #covar += activation[0].T @ other_activation[0]
                    activation = []
                covar /= total
                print("done with covariance_calc")
                hook_handle_1.remove()
                hook_handle_2.remove()
                u, s, v = torch.linalg.svd(covar, full_matrices=False)
                align = u @ v
                new_module.register_buffer("input_align", align)
                new_sd['input_align'] = align
                def return_hook():
                    def hook(mod, inputs):
                        shape = inputs[0].shape
                        inputs_permute = inputs[0].permute(1,0,2,3).reshape(inputs[0].shape[1], -1)
                        reshape = (mod.input_align@inputs_permute).reshape(shape[1],
                                shape[0], shape[2],
                                shape[3]).permute(1, 0, 2, 3)
                        return reshape 
                    return hook
                hook_handle_pre_forward = new_module.register_forward_pre_hook(return_hook())

                #else:
                #    new_weight = new_sd['weight']
                #    new_weight = torch.moveaxis(new_weight, source=1,
                #            destination=-1)
                #    print(align.shape)
                #    print(new_weight.shape)
                #    new_weight = new_weight@align
                #    new_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                #            destination=1)        
            #Set the new weights in the generated model.
            #NOTE: this an intermediate model, as sampling the j-th layer means that the j+1-th layer is no longer aligned.
            #As such, if evaluated as is, its accuracy would be that of a random guess.
            new_module.load_state_dict(new_sd)
            setattr(new_model, n1, new_module)
        if isinstance(m1, nn.Linear):
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
            for batch_idx, (inputs, targets) in enumerate(train_loader):
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
            u, s, v = torch.linalg.svd(covar, full_matrices=False)
            align = u @ v
            new_weight = new_sd['weight']
            new_weight = torch.moveaxis(new_weight, source=1,
                    destination=-1)
            new_weight = new_weight@align
            new_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                    destination=1)        
            new_module.load_state_dict(new_sd)
            setattr(new_model, n1, new_module)
 
 
            #print("linear")
        if isinstance(m1,nn.BatchNorm2d):
            print("BatchieNormie")
            m1.train()
            m2.train()
            m1.reset_running_stats()
            m2.reset_running_stats()
            #bn_hook = 
            handel_1 = m1.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
            handel_2 = m2.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
            m1.to(device)
            m2.to(device)
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                #outputs1 = net(inputs)
                #outputs2 = net_new(inputs)
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
 



@torch.no_grad()
def rainbow_sampling(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            rainbow_sampling(module)
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
        if isinstance(module, nn.Conv2d) or (isinstance(module, nn.Linear) and module.out_features != 1000):
            ## simple module
            if isinstance(module, nn.Conv2d):
                new_module = nn.Conv2d(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride, padding=module.padding, 
                        bias=True if module.bias is not None else False,
                        groups=module.groups).to(device)
                nn.init.trunc_normal_(new_module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                new_module = nn.Linear(module.in_features, module.out_features,
                        bias=True if module.bias is not None else False).to(device)
                nn.init.trunc_normal_(new_module.weight, std=0.02)

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


def reset_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            reset_bn(module)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                module.train()
 
def turn_off_grads(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            turn_off_grads(module)
        else:
            if isinstance(module, nn.Linear) and module.out_features == 1000:
                grad=True
            else:
                grad=False
            for param in module.parameters():
                param.requires_grad = grad


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
import time 
print("sampling")
s = time.time()
#recurse_layersrainbow_sampling(net)
recurse_layers(net, net_new)
#reset_bn(net)
net.train()
turn_off_grads(net)
#with torch.no_grad():
#    for i, (images, target) in enumerate(train_loader):
#        images = images.cuda(non_blocking=True)
#        output = net(images)

print(time.time() - s)
print('validating')
validate(val_loader, net, criterion)
s = time.time()
print("training")
optimizer = torch.optim.SGD(filter(lambda param:param.requires_grad,
    net.parameters()), 0.0001,
                                momentum=.9,
                                weight_decay=1e-4)

train(train_loader, net, criterion, optimizer, 0, device)

print("validating")
validate(val_loader, net, criterion)
