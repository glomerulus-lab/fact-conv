import torch.nn as nn
import torch
import torch.nn.functional as F
import LearnableCov


class V1_MNIST(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(V1_MNIST, self).__init__()
        self.lc_layer = \
            LearnableCov.FactConv2d(in_channels=1, out_channels=hidden_dim,
                                       kernel_size=7, stride=1, padding=3, 
                                       bias=bias) 
        self.lc_layer2 = \
            LearnableCov.FactConv2d(in_channels=hidden_dim,
                                       out_channels=hidden_dim, kernel_size=7,
                                       stride=1, padding=3, 
                                       bias=bias)
        self.clf = nn.Linear((1 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + \
                             (hidden_dim * (8 ** 2)), 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = None
        
        # if bias==True:
        #     self.lc_layer.bias.requires_grad = False
        #     self.lc_layer2.bias.requires_grad = False
        
    def forward(self, x):  #[128, 1, 28, 28]
        h1 = self.relu(self.lc_layer(self.bn(x)))  
        h2 = self.relu(self.lc_layer2(h1))  
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)  
        x_pool = self.bn0(pool(x))  
        h1_pool = self.bn1(pool(h1))  
        h2_pool = self.bn2(pool(h2))  
        
        x_flat = x_pool.view(x_pool.size(0), -1)  
        h1_flat = h1_pool.view(h1_pool.size(0),  -1)  
        h2_flat = h2_pool.view(h2_pool.size(0), -1) 
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  
        
        beta = self.clf(concat) 
        return beta

class V1_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias,
            freeze_spatial, freeze_channel, spatial_init, seed=None):
        super().__init__()
        self.lc_layer = \
                LearnableCov.FactConv2d(in_channels=3, out_channels=hidden_dim,
                        kernel_size=7, stride=1, padding=3, bias=bias)
        self.lc_layer2 = \
                LearnableCov.FactConv2d(in_channels=hidden_dim + 3,
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

class Factorized_V1_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(V1_CIFAR10, self).__init__()
        self.lc_layer = \
                LearnableCov.FactConv2d(in_channels=3, out_channels=hidden_dim,
                        kernel_size=7, stride=1, padding=3, bias=bias)
        self.lc_layer2 = \
                LearnableCov.FactConv2d(in_channels=hidden_dim + 3,
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

class V1_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(V1_CIFAR100, self).__init__()
        self.lc_layer = \
                LearnableCov.FactConv2d(in_channels=3, out_channels=hidden_dim,
                        kernel_size=7, stride=1, padding=3, bias=bias)
        self.lc_layer2 = \
                LearnableCov.FactConv2d(in_channels=hidden_dim + 3,
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

        LearnableCov.V1_init(self.lc_layer, size, spatial_freq, center, scale1, bias, seed)
        self.lc_layer.tri2_vec.requires_grad=False

        LearnableCov.V1_init(self.lc_layer2, size, spatial_freq, center, scale1, bias, seed)
        self.lc_layer2.tri2_vec.requires_grad=False
        

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


class SimpleLearnableNetwork(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(SimpleLearnableNetwork, self).__init__()
        self.layer = LearnableCov.Conv2d(in_channels=1, out_channels=hidden_dim,
                kernel_size=7, stride=1, padding=3, bias=bias)
        self.layer2 = LearnableCov.Conv2d(in_channels=hidden_dim,
                out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                bias=bias)
        self.relu = nn.ReLU()
        self.clf = nn.Linear(1344, 100)

    def forward(self, x):
        x1 = self.relu(self.layer(x))
        x2 = self.relu(self.layer2(x1))

        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)
        x_pool = pool(x)
        x1_pool = pool(x1)
        x2_pool = pool(x2)

        x_flat = x_pool.view(x_pool.size(0), -1)
        x1_flat = x1_pool.view(x1_pool.size(0), -1)
        x2_flat = x2_pool.view(x2_pool.size(0), -1)

        concat = torch.cat((x_flat, x1_flat, x2_flat), 1)

        return self.clf(concat)


class SimpleUnlearnableNetwork(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(SimpleUnlearnableNetwork, self).__init__()
        self.layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim,
                kernel_size=7, stride=1, padding=3, bias=bias)
        self.layer2 = nn.Conv2d(in_channels=hidden_dim,
                out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                bias=bias)
        self.relu = nn.ReLU()
        self.clf = nn.Linear(1344, 100)

    def forward(self, x):
        x1 = self.relu(self.layer(x))
        x2 = self.relu(self.layer2(x1))

        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)
        x_pool = pool(x)
        x1_pool = pool(x1)
        x2_pool = pool(x2)

        x_flat = x_pool.view(x_pool.size(0), -1)
        x1_flat = x1_pool.view(x1_pool.size(0), -1)
        x2_flat = x2_pool.view(x2_pool.size(0), -1)

        concat = torch.cat((x_flat, x1_flat, x2_flat), 1)

        return self.clf(concat)
