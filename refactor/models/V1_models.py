import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights
import gc
import LearnableCov

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

class V1_CIFAR10(nn.Module):
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
        h = self.bn_h1(h)
        h = torch.cat((self.relu(self.v1_layer2(h)), smooth(h)), 1)
        h = self.bn_h2(h)
        h = flatten(pool(h))
        return self.clf(h)
 
        
class Rand_Scat_Block(nn.Module):
    def __init__(self, in_chan, num_filt, size, spatial_freq,
                 kernel_size=7, stride=1, padding=3, scale=None, bias=True, seed=None):
        super().__init__()

        out_chan = in_chan + num_filt

        self.v1 = nn.Conv2d(in_channels=in_chan, out_channels=num_filt,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            scale=scale, bias=bias)
        self.bn = nn.BatchNorm2d(num_filt)
        self.relu = nn.ReLU()

        # V1 params
        if scale is None:
            scale = 1 / (in_chan * np.prod(kernel_size))
        center = ((kernel_size - 1) / 2, (kernel_size - 1) / 2)

        # init weights
        V1_init(self.v1, size, spatial_freq, center, scale, bias, seed)
        self.v1.weight.requires_grad = False
        if bias:
            self.v1.bias.requires_grad = False

    def forward(self, x):
        smooth = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        h = self.relu(self.v1(x))
        concat = torch.cat((h, smooth(x)), 1) # concatenate with smoothed input
        return self.bn(concat)

class Rand_Scat_CIFAR10(nn.Module):
    pass

class Learned_Rand_Scat_CIFAR10(nn.Module):
    def __init__(self, num_filt, size, spatial_freq, scale, bias, seed=None):
        super().__init__()

        # channel dimensions
        dims = [3, 64, 128]
        
        # fixed feature layers
        self.v1_layer = nn.Conv2d(in_channels=dims[0], out_channels=num_filt,
                                  kernel_size=7, stride=1, padding=3, bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=dims[1], out_channels=num_filt,
                                   kernel_size=7, stride=1, padding=3, bias=bias)
        self.relu = nn.ReLU()
        
        # unsupervised layers
        self.bn_x = nn.BatchNorm2d(3)
        self.bn_h1 = nn.LazyBatchNorm2d()
        self.bn_h2 = nn.LazyBatchNorm2d()

        # supervised layers
        self.L1 = nn.LazyConv2d(out_channels=dims[1], kernel_size=1, bias=False)
        self.L2 = nn.LazyConv2d(out_channels=dims[2], kernel_size=1, bias=False)
        self.clf = nn.LazyLinear(10)

        # init fixed weights
        scale1 = 1 / (3 * 7 * 7)
        scale2 = 1 / (num_filt * 7 * 7)
        center = (3., 3.)
        
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
# OLD 
#     def forward(self, x):  
#         h1 = self.relu(self.v1_layer(self.bn_x(x))) 
#         h2 = self.relu(self.v1_layer2(self.bn_h1(h1)))
        
#         pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)  
#         x_pool = self.bn0(pool(x)) 
#         h1_pool = self.bn1(pool(h1))  
#         h2_pool = self.bn2(pool(h2))  
        
#         x_flat = x_pool.view(x_pool.size(0), -1)   #view
#         h1_flat = h1_pool.view(h1_pool.size(0),  -1)  #view
#         h2_flat = h2_pool.view(h2_pool.size(0), -1)  #view

        
#         concat = torch.cat((x_flat, h1_flat, h2_flat), 1) 

    def forward(self, x):
        # methods
        smooth = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        flatten = nn.Flatten()

        # do pass
        x = self.bn_x(x)
        h1 = self.relu(self.v1_layer(x))
        h1 = torch.cat((h1, smooth(x)), 1)
        h1 = self.bn_h1(h1) # chan: 3 + num_filt
        h2 = self.relu(self.v1_layer2(self.L1(h1)))
        h2 = torch.cat((h2, smooth(h1)), 1) # chan: 3 + num_filt + dims[1]
        h2 = self.bn_h2(h2)

        concat = flatten(pool(h2))

        beta = self.clf(concat)
       
        return beta


class V1_CIFAR100(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(V1_CIFAR100, self).__init__()
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
    
class V1_MNIST(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(V1_MNIST, self).__init__()
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
        h = self.bn_h1(h)
        h = torch.cat((self.relu(self.v1_layer2(h)), smooth(h)), 1)
        h = self.bn_h2(h)
        h = flatten(pool(h))
        return h

