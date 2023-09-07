import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
#sys.path.insert(0, '/home/mila/v/vivian.white/structured-random-features/')
from src.models.init_weights import V1_init, classical_init, V1_weights
import gc
#from pytorch_memlab import LineProfiler, MemReporter, profile, set_target_gpu
#set_target_gpu(1)

def train(model, model_init, penalty, device, train_loader, optimizer, epoch):
    model.train()
    avg_cost = 0.
    avg_loss = 0.
    avg_reg = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        reg = regularizer(model, model_init)
        loss = F.cross_entropy(output, target)
        cost = loss + penalty * reg
        # print(f"{batch_idx}: " \
        #       f"cost {cost:.3e} = " \
        #       f"loss {loss:.3e} + " \
        #       f"regularization {reg:.3e} * penalty {penalty:.1e}")
        avg_cost += float(cost)
        avg_loss += float(loss)
        avg_reg += float(reg)
        cost.backward()
        optimizer.step()

    avg_cost /= len(train_loader)
    avg_loss /= len(train_loader)
    avg_reg /= len(train_loader)
    print(f"cost {avg_cost:.3e} = " \
          f"loss {avg_loss:.3e} + " \
          f"regularization {avg_reg:.3e} * penalty {penalty:.1e}")


def regularizer(model, model_init):
    cost = 0.
    for name, new_param in model.scattering_layers.state_dict().items():
        if 'weight' in name:
            init_param = model_init.scattering_layers.state_dict()[name]
            new_param = model.scattering_layers.state_dict()[name]
            # print(type(new_param))
            cost += torch.mean( (new_param - init_param) ** 2 )
    return cost

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('Test Epoch: {}\t Avg Loss: {:.4f}\t Accuracy: {:.2f}%'.format(
        epoch, test_loss, accuracy))
    return test_loss, accuracy

class BN_V1_V1_LinearLayer_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(BN_V1_V1_LinearLayer_CIFAR10, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                                  kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim,
                                   out_channels=hidden_dim,
                                   kernel_size=7, stride=1, padding=3, bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) +
                             (hidden_dim * (8 ** 2)), 10)
        self.relu = nn.ReLU()
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.scattering_layers = nn.ModuleList([self.v1_layer, self.v1_layer2])

        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = (3., 3.)
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        self.v1_layer.weight.requires_grad = True
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        self.v1_layer2.weight.requires_grad = True
        
        # if bias==True:
        #     self.v1_layer.bias.requires_grad = False
        #     self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  
        h1 = self.relu(self.v1_layer(self.bn_x(x))) 
        h2 = self.relu(self.v1_layer2(self.bn_h1(h1)))
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)  
        x_pool = self.bn0(pool(x)) 
        h1_pool = self.bn1(pool(h1))  
        h2_pool = self.bn2(pool(h2))  
        
        x_flat = x_pool.view(x_pool.size(0), -1)   #view
        h1_flat = h1_pool.view(h1_pool.size(0),  -1)  #view
        h2_flat = h2_pool.view(h2_pool.size(0), -1)  #view

        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1) 
        
        beta = self.clf(concat)
        return beta
    

class BN_V1_V1_LinearLayer_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(BN_V1_V1_LinearLayer_CIFAR100, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim)
        
        self.scattering_layers = nn.ModuleList([self.v1_layer, self.v1_layer2])

        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        center = None
        
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        #self.v1_layer.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        #self.v1_layer2.weight.requires_grad = False
        
        #if bias==True:
        #    self.v1_layer.bias.requires_grad = False
        #    self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  #[128, 3, 32, 32]
        h1 = self.relu(self.v1_layer(self.bn(x)))  
        h2 = self.relu(self.v1_layer2(self.bn_h1(h1))) 
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)  
        x_pool = self.bn0(pool(x))  
        h1_pool = self.bn1(pool(h1)) 
        h2_pool = self.bn2(pool(h2))
        
        x_flat = x_pool.view(x_pool.size(0), -1)  
        h1_flat = h1_pool.view(h1_pool.size(0),  -1) 
        h2_flat = h2_pool.view(h2_pool.size(0), -1)  
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  
        
        beta = self.clf(concat) 
        return beta
    
class BN_V1_V1_LinearLayer_MNIST(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(BN_V1_V1_LinearLayer_MNIST, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.clf = nn.Linear((1 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
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
        
    def forward(self, x):  #[128, 1, 28, 28]
        h1 = self.relu(self.v1_layer(self.bn(x)))  
        h2 = self.relu(self.v1_layer2(h1))  
        
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
    
class Scattering_V1_MNIST(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(Scattering_V1_MNIST, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
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
        
    def forward(self, x):  #[128, 1, 28, 28]
        h1 = self.relu(self.v1_layer(self.bn(x)))  
        h2 = self.relu(self.v1_layer2(h1))  
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)  
        x_pool = self.bn0(pool(x))  
        h1_pool = self.bn1(pool(h1))  
        h2_pool = self.bn2(pool(h2))  
        
        x_flat = x_pool.view(x_pool.size(0), -1)  
        h1_flat = h1_pool.view(h1_pool.size(0),  -1)  
        h2_flat = h2_pool.view(h2_pool.size(0), -1) 
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  
        return concat

class Scattering_V1_celeba(nn.Module):
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
        # methods
        # smooth = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        flatten = nn.Flatten()

        x = self.bn_x(x)
        h = torch.cat((self.relu(self.v1_layer(x)), smooth(x)), 1)
        gc.collect()
        h = self.bn_h1(h)
        h = torch.cat((self.relu(self.v1_layer2(h)), smooth(h)), 1)
        h = self.bn_h2(h)
        h = flatten(pool(h))
        return h

