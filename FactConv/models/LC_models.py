import torch.nn as nn
import torch
import torch.nn.functional as F
from FactConv.conv_modules import FactConv2d
from FactConv.V1_covariance import V1_init

class LC_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init, seed=None):
        super().__init__()
        self.lc_layer = \
                FactConv2d(in_channels=3, out_channels=hidden_dim,
                        kernel_size=7, stride=1, padding=3, bias=bias)
        self.lc_layer2 = \
                FactConv2d(in_channels=hidden_dim + 3,
                        out_channels=hidden_dim, kernel_size=7,
                        stride=1, padding=3, bias=bias)
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
            V1_init(self.lc_layer, size, spatial_freq, center, scale1, bias, seed)
            V1_init(self.lc_layer2, size, spatial_freq, center, scale2, bias, seed)
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



class LC_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init, seed=None):
        super(LC_CIFAR100, self).__init__()
        self.lc_layer = \
                FactConv2d(in_channels=3, out_channels=hidden_dim,
                        kernel_size=7, stride=1, padding=3, bias=bias)
        self.lc_layer2 = \
                FactConv2d(in_channels=hidden_dim + 3,
                        out_channels=hidden_dim, kernel_size=7,
                        stride=1, padding=3, bias=bias)
        self.relu = nn.ReLU()

        # unsupervised layers
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim + 3)
        self.bn_h2 = nn.BatchNorm2d(hidden_dim * 2 + 3)

        # supervised layers
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2))\
                + (hidden_dim * (8 ** 2)), 100)

        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = (3., 3.,)
        #center = None

        if spatial_init == 'V1':
            V1_init(self.lc_layer, size, spatial_freq, center, scale1, bias, seed)
            V1_init(self.lc_layer2, size, spatial_freq, center, scale2, bias, seed)
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
