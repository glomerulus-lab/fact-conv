import torch.nn as nn
import torch
import torch.nn.functional as F
import LearnableCov
from ConvModules import FactConv2dPostExp, FactConv2dPreExp

from resnet import ResNet18, ResNet18_Class100
from factnet import FactNet18, FactNet18_Class100


def replace_layers_agnostic(model, scale, num_classes):
    prev_outch = 0
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_agnostic(module,scale,num_classes)
        if isinstance(module, nn.Conv2d):
            if module.in_channels == 3:
                in_scale = 1 
            else:
                in_scale = scale
            ## simple module
            new_module = LearnableCov.FactConv2d(
                    in_channels=module.in_channels*in_scale,
                    out_channels=module.out_channels*scale,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=module.bias)
            new_module.tri1_vec = nn.Parameter(new_module.tri1_vec * scale)
            setattr(model, n, new_module)
            prev_outch = new_module.out_channels
        if isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(prev_outch)
            setattr(model, n, new_module)
        if isinstance(module, nn.Linear):
            new_module = nn.Linear(512 * scale, num_classes)
            setattr(model, n, new_module)


class Replacenet_V1_CIFAR10(nn.Module):
    def __init__(self, param_scale, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init,
            replaceModule=LearnableCov.FactConv2d, seed=None):
        super().__init__()
        self.resnet = ResNet18()
        print("PARAM SCALE: ", param_scale)
        replace_layers_agnostic(self.resnet, param_scale, 10)

        firstConv = True
        for c, m in enumerate(self.resnet.modules()):
            if isinstance(m, LearnableCov.FactConv2d) or isinstance(m,
                    FactConv2dPostExp) or isinstance(m, FactConv2dPreExp):
                scale = 1/(m.in_channels*m.kernel_size[0]*m.kernel_size[1])
                center = (int(m.kernel_size[0]/2), int(m.kernel_size[1]/2))

                if spatial_init == 'V1':
                    if m.kernel_size[0] == 1 and m.kernel_size[1] == 1:
                        pass
                    else:
                        LearnableCov.V1_init(m, size, spatial_freq, center, scale, bias, seed)
                    if firstConv:
                        print("V1 spatial init")
                else:
                    if firstConv:
                        print("Default spatial init")
                
                if freeze_spatial == True:
                    m.tri2_vec.requires_grad=False
                    if firstConv:
                        print("Freeze spatial vec")
                else:
                    if m.kernel_size[0] == 1 and m.kernel_size[1] == 1:
                        m.tri2_vec.requires_grad=False
                    else:
                        m.tri2_vec.requires_grad=True
                        if firstConv:
                            print("Learnable spatial vec")

                if freeze_channel == True:
                    m.tri1_vec.requires_grad=False
                    if firstConv:
                        print("Freeze channel vec")
                else:
                    m.tri1_vec.requires_grad=True
                    if firstConv:
                        print("Learnable channel vec")
                firstConv = False


    def forward(self, x):
        return self.resnet(x)


class Factnet_V1_CIFAR10(nn.Module):
    def __init__(self, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init, seed=None):
        super().__init__()
        self.resnet = FactNet18()

        firstConv = True
        for c, m in enumerate(self.resnet.modules()):
            if isinstance(m, LearnableCov.FactConv2d):
                scale = 1/(m.in_channels*m.kernel_size[0]*m.kernel_size[1])
                center = (int(m.kernel_size[0]/2), int(m.kernel_size[1]/2))

                if spatial_init == 'V1':
                    LearnableCov.V1_init(m, size, spatial_freq, center, scale, bias, seed)
                    if firstConv:
                        print("V1 spatial init")
                else:
                    if firstConv:
                        print("Default spatial init")
                
                if freeze_spatial == True:
                    m.tri2_vec.requires_grad=False
                    if firstConv:
                        print("Freeze spatial vec")
                else:
                    m.tri2_vec.requires_grad=True
                    if firstConv:
                        print("Learnable spatial vec")

                if freeze_channel == True:
                    m.tri1_vec.requires_grad=False
                    if firstConv:
                        print("Freeze channel vec")
                else:
                    m.tri1_vec.requires_grad=True
                    if firstConv:
                        print("Learnable channel vec")
                firstConv = False


    def forward(self, x):
        return self.resnet(x)




def PostExp(size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init):
    return Replacenet_V1_CIFAR10(size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init, FactConv2dPostExp)


def PreExp(param_scale, size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init):
    return Replacenet_V1_CIFAR10(param_scale, size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init, FactConv2dPreExp)



class Replacenet_V1_CIFAR100(nn.Module):
    def __init__(self, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init,
            replaceModule=LearnableCov.FactConv2d, seed=None):
        super().__init__()
        self.resnet = ResNet18_Class100()
        scale = 2
        replace_layers_agnostic(self.resnet, scale, 100)
#        def replace_layers(model):
#            for n, module in model.named_children():
#                if len(list(module.children())) > 0:
#                    ## compound module, go inside it
#                    replace_layers(module)
#                if isinstance(module, nn.Conv2d):
#                    ## simple module
#                    new_module = replaceModule(in_channels=module.in_channels,
#                                out_channels=module.out_channels,
#                                kernel_size=module.kernel_size,
#                                stride=module.stride, padding=module.padding,
#                                bias=module.bias)
#                    setattr(model, n, new_module)
#
#        replace_layers(self.resnet)
        
        firstConv = True
        for c, m in enumerate(self.resnet.modules()):
            if isinstance(m, LearnableCov.FactConv2d) or isinstance(m,
                    FactConv2dPostExp) or isinstance(m, FactConv2dPreExp):
                scale = 1/(m.in_channels*m.kernel_size[0]*m.kernel_size[1])
                center = (int(m.kernel_size[0]/2), int(m.kernel_size[1]/2))

                if spatial_init == 'V1':
                    if m.kernel_size[0] == 1 and m.kernel_size[1] == 1:
                        pass
                    else:
                        LearnableCov.V1_init(m, size, spatial_freq, center, scale, bias, seed)
                    if firstConv:
                        print("V1 spatial init")
                else:
                    if firstConv:
                        print("Default spatial init")
                
                if freeze_spatial == True:
                    m.tri2_vec.requires_grad=False
                    if firstConv:
                        print("Freeze spatial vec")
                else:
                    if m.kernel_size[0] == 1 and m.kernel_size[1] == 1:
                        m.tri2_vec.requires_grad=False
                    else:
                        m.tri2_vec.requires_grad=True
                        if firstConv:
                            print("Learnable spatial vec")

                if freeze_channel == True:
                    m.tri1_vec.requires_grad=False
                    if firstConv:
                        print("Freeze channel vec")
                else:
                    m.tri1_vec.requires_grad=True
                    if firstConv:
                        print("Learnable channel vec")
                firstConv = False


    def forward(self, x):
        return self.resnet(x)



def PostExp_Class100(size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init):
    return Replacenet_V1_CIFAR100(size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init, FactConv2dPostExp)


def PreExp_Class100(size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init):
    return Replacenet_V1_CIFAR100(size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init, FactConv2dPreExp)



class V1_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init,
            replaceModule=LearnableCov.FactConv2d, seed=None):
        super().__init__()
        self.lc_layer = \
                LearnableCov.FactConv2d(in_channels=3, out_channels=hidden_dim,
                        kernel_size=7, stride=1, padding=3, bias=bias)
        self.lc_layer2 = \
                LearnableCov.FactConv2d(in_channels=hidden_dim + 3,
                        out_channels=hidden_dim, kernel_size=7,
                        stride=1, padding=3, bias=bias)
        def replace_layers(model):
            for n, module in model.named_children():
                if len(list(module.children())) > 0:
                    ## compound module, go inside it
                    replace_layers(module)
                if isinstance(module, nn.Conv2d):
                    ## simple module
                    new_module = replaceModule(in_channels=module.in_channels,
                                out_channels=module.out_channels,
                                kernel_size=module.kernel_size,
                                stride=module.stride, padding=module.padding,
                                bias=module.bias)
                    setattr(model, n, new_module)

        replace_layers(self)

        self.relu = nn.ReLU()

        # unsupervised layers
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim + 3)
        self.bn_h2 = nn.BatchNorm2d(hidden_dim * 2 + 3)

        # supervised layers
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2))\
                + (hidden_dim * (8 ** 2)), 10)

        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = (3., 3.,)

        if spatial_init == 'V1':
            LearnableCov.V1_init(self.lc_layer, size, spatial_freq, center, scale1, bias, seed)
            LearnableCov.V1_init(self.lc_layer2, size, spatial_freq, center, scale2, bias, seed)
            print("V1 spatial init")
        else:
            print("Default spatial init")
        
        if freeze_spatial == True:
            self.lc_layer.tri2_vec.requires_grad=False
            self.lc_layer2.tri2_vec.requires_grad=False
            print("Freeze spatial vec")
        else:
            print("Learnable spatial vec")

        if freeze_channel == True:
            self.lc_layer.tri1_vec.requires_grad=False
            self.lc_layer2.tri1_vec.requires_grad=False
            print("Freeze channel vec")
        else:
            print("Learnable channel vec")
    
    def forward(self, x):
        # methods
        smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        flatten = nn.Flatten()

        x = self.bn_x(x)
        h = torch.cat((self.relu(self.lc_layer(x)), smooth(x)), 1)
        h = self.bn_h1(h) 
        h = torch.cat((self.relu(self.lc_layer2(h)), smooth(h)), 1)
        h = self.bn_h2(h)
        h = flatten(pool(h))
        return self.clf(h)


def PostExp_Vivian(hidden_dim, size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init):
    net = V1_CIFAR10(hidden_dim, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init, FactConv2dPostExp)
    return net


def PreExp_Vivian(hidden_dim, size, spatial_freq, scale, bias, freeze_spatial, freeze_channel, spatial_init):
    net = V1_CIFAR10(hidden_dim, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init, FactConv2dPreExp)
    return net
