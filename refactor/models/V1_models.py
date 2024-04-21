import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/mila/v/vivian.white/structured-random-features')
from src.models.init_weights import V1_init, classical_init, V1_weights
import gc

class V1_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias=False, seed=None):
        super().__init__()

        self.bn_x = nn.BatchNorm2d(3)
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                                  kernel_size=7, stride=1, padding=3, bias=bias) 
        self.relu = nn.ReLU()
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim + 3)
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim + 3, out_channels=hidden_dim,
                                   kernel_size=7, stride=1, padding=3, bias=bias)
        
        self.bn_h2 = nn.BatchNorm2d(hidden_dim * 2 + 3)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        self.flatten = nn.Flatten()
        self.clf = nn.Linear((8 ** 2) * (hidden_dim * 2 + 3), 10)
        
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = (3., 3.)
        
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        self.v1_layer2.weight.requires_grad = False
        
        if bias:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):
        x = self.bn_x(x)
        h = torch.cat((self.relu(self.v1_layer(x)), self.smooth(x)), 1)
        h = self.bn_h1(h)
        h = torch.cat((self.relu(self.v1_layer2(h)), self.smooth(h)), 1)
        h = self.bn_h2(h)
        h = self.flatten(self.pool(h))
        return self.clf(h)
 
        

class V1_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(V1_CIFAR100, self).__init__()
        
        self.bn_x = nn.BatchNorm2d(3)
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.relu = nn.ReLU()
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim)
        
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)  
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 100)
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = None
        
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  #[128, 3, 32, 32]
        h1 = self.relu(self.v1_layer(self.bn_x(x)))  
        h2 = self.relu(self.v1_layer2(self.bn_h1(h1))) 
        
        x_pool = self.bn0(self.pool(x))  
        h1_pool = self.bn1(self.pool(h1)) 
        h2_pool = self.bn2(self.pool(h2))
        
        x_flat = x_pool.view(x_pool.size(0), -1)  
        h1_flat = h1_pool.view(h1_pool.size(0),  -1) 
        h2_flat = h2_pool.view(h2_pool.size(0), -1)  
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  
        
        beta = self.clf(concat) 
        return beta
    
