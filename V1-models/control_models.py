import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) 
        loss.backward()
        optimizer.step()
             
                      
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('Test Epoch: {}\t Avg Loss: {:.4f}\t Accuracy: {:.2f}%'.format(
        epoch, test_loss, accuracy))

    return test_loss, accuracy


def train_scatter(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        print("output shape: ", output.shape)
        loss = F.cross_entropy(output, target)
        print("target shape: ", target.shape)
        loss.backward()
        optimizer.step()

def test_scatter(model, device, test_loader, epoch, scattering):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('Test Epoch: {}\t Avg Loss: {:.4f}\t Accuracy: {:.2f}%'.format(
        epoch, test_loss, accuracy))
    
    return test_loss, accuracy

class Gaussian_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, bias, scale, seed=None):
        super(Gaussian_CIFAR10, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                                  bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 10)
        self.relu = nn.ReLU()
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        
        classical_init(self.v1_layer, scale1, bias, seed)
        classical_init(self.v1_layer2, scale2, bias, seed)
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):
        h1 = self.relu(self.v1_layer(self.bn_x(x))) 
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
    
class Gaussian_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, bias, scale, seed=None):
        super(Gaussian_CIFAR100, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                                  bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        
        classical_init(self.v1_layer, scale1, bias, seed)
        classical_init(self.v1_layer2, scale2, bias, seed)
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x): 
        h1 = self.relu(self.v1_layer(self.bn(x))) 
        h2 = self.relu(self.v1_layer2(h1))  
        
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
    
class Gaussian_MNIST(nn.Module):
    def __init__(self, hidden_dim, bias, scale, seed=None):
        super(Gaussian_MNIST, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                                  bias=bias)
        self.clf = nn.Linear((1 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (hidden_dim * 7 * 7)
        
        classical_init(self.v1_layer, scale1, bias, seed)
        classical_init(self.v1_layer2, scale2, bias, seed)
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  
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
    
class Uniform_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, bias, seed=None):
        super(Uniform_CIFAR10, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                                  bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 10)
        self.relu = nn.ReLU()
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.BatchNorm2d(hidden_dim)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  
        h1 = self.relu(self.v1_layer(self.bn_x(x))) 
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
    
class Uniform_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, bias, seed=None):
        super(Uniform_CIFAR100, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                                  bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x): 
        h1 = self.relu(self.v1_layer(self.bn(x)))  
        h2 = self.relu(self.v1_layer2(h1)) 
        
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
    
    
class Uniform_MNIST(nn.Module):
    def __init__(self, hidden_dim, bias, seed=None):
        super(Uniform_MNIST, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3,
                                  bias=bias)
        self.clf = nn.Linear((1 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  
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
    
class Scattering_Linear_CIFAR10(nn.Module):
    def __init__(self):
        super(Scattering_Linear_CIFAR10, self).__init__()
        self.clf = nn.Linear(3264, 10)
       
    def forward(self, x): 
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x
    
class Scattering_Linear_CIFAR100(nn.Module):
    def __init__(self):
        super(Scattering_Linear_CIFAR100, self).__init__()
        self.clf = nn.Linear(3264, 100)
       
    def forward(self, x): 
        x = x.view(x.size(0), -1) 
        x = self.clf(x)
        return x
    
class Scattering_Linear_MNIST(nn.Module):
    def __init__(self):
        super(Scattering_Linear_MNIST, self).__init__()
        self.clf = nn.Linear(833, 10)
       
    def forward(self, x): 
        x = x.view(x.size(0), -1) 
        x = self.clf(x)
        return x
    
