import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights

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

class BN_V1_V1_LinearLayer_CIFAR10(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(BN_V1_V1_LinearLayer_CIFAR10, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.clf = nn.Linear((3 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)
        self.bn0 = nn.BatchNorm2d(3)
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
        
    def forward(self, x):  
        h1 = self.relu(self.v1_layer(self.bn(x))) 
        h2 = self.relu(self.v1_layer2(h1))  
        
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
    
class Generator_V1_MNIST(nn.Module):
    def __init__(self, hidden_dim, output_dim, size, spatial_freq, scale, bias, seed=None):
        super(Generator_V1_MNIST, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.clf = nn.Linear((1 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), output_dim)
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

class Generator_V1_celeba(nn.Module):
    def __init__(self, hidden_dim, output_dim, size, spatial_freq, scale, bias, seed=None):
        super(Generator_V1_celeba, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        self.clf = nn.Linear(221067, output_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)
        self.bn0 = nn.BatchNorm2d(3)
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
        
    def forward(self, x):  #[2, 3, 218, 178]
        #print("x: ", x.shape)
        #print("bn: ", self.bn(x).shape)
        h1 = self.relu(self.v1_layer(self.bn(x)))  
        h2 = self.relu(self.v1_layer2(h1))  
        
        #print("h1: ", h1.shape)
        #print("h2: ", h2.shape)
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)  
        x_pool = self.bn0(pool(x))  
        h1_pool = self.bn1(pool(h1))  
        h2_pool = self.bn2(pool(h2))  

        
        #print("x pool ", x_pool.shape)
        #print("h1_pool: ", h1_pool.shape)
        #print("h2 pool: ", h2_pool.shape)
        
        x_flat = x_pool.view(x_pool.size(0), -1)  
        h1_flat = h1_pool.view(h1_pool.size(0),  -1)  
        h2_flat = h2_pool.view(h2_pool.size(0), -1) 
        
        #print("x flat: ", x_flat.shape)
        #print("h1 flat: ", h1_flat.shape)
        #print("h2 flat: ", h2_flat.shape)
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  
        
        beta = self.clf(concat) 
        return beta

