import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
#sys.path.insert(0, '/home/mila/v/vivian.white/structured-random-features/')
sys.path.insert(0, '/home/harri267/work/structured-random-features/')
from src.models.init_weights import V1_init


class Sequential_ThreeLayer_CIFAR10(nn.Module):
    def __init__(self, hidden_dim=100, bias=False):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
                Fact_Concat(in_channels=3, 
                    hidden_dim=hidden_dim,
                    bn_dim=hidden_dim+3,
                    bias=bias),
                Fact_Concat(in_channels=hidden_dim + 3, 
                    hidden_dim=hidden_dim, 
                    bn_dim=hidden_dim * 2 + 3,
                    bias=bias),
                )
        self.clf = nn.Linear((8 ** 2) * (hidden_dim * 2 + 3), 10)

    def forward(self, x):
        h = self.flatten(self.pool(self.layers(x)))
        h = self.clf(h)
        return h

class Fact_Concat(nn.Module):
    def __init__(self, in_channels, hidden_dim, bn_dim, bias, seed=None):
        super().__init__()

        self.v1_layer = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim,
                kernel_size=7, stride=1, padding=3, bias=bias)
        self.b1 = nn.BatchNorm2d(bn_dim)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = torch.cat((self.relu(self.v1_layer(x)), self.smooth(x)), 1)
        h = self.b1(h)
        return h

class ThreeLayer_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super().__init__()

        # fixed feature layers
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                                  kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim + 3, out_channels=hidden_dim,
                                   kernel_size=7, stride=1, padding=3, bias=bias)
        self.relu = nn.ReLU()
        
        # unsupervised layers
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim + 3)
        self.bn_h2 = nn.BatchNorm2d(hidden_dim * 2 + 3)

        # supervised layers
        self.clf = nn.Linear((8 ** 2) * (hidden_dim * 2 + 3), 10)
        
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = (3., 3.)
        
#        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
#        self.v1_layer.weight.requires_grad = False
        
#        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
#        self.v1_layer2.weight.requires_grad = False
        
        if bias:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):
        # methods
        # smooth = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        flatten = nn.Flatten()
        
        #x = self.bn_x(x)
        h = torch.cat((self.relu(self.v1_layer(x)), smooth(x)), 1)
        h = self.bn_h1(h)
        h = torch.cat((self.relu(self.v1_layer2(h)), smooth(h)), 1)
        h = self.bn_h2(h)
        h = flatten(pool(h))
        return self.clf(h)
