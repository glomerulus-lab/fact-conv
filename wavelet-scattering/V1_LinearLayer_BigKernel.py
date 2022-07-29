"""
Classification on CIFAR10 (ResNet)
==================================
Based on pytorch example for CIFAR10
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
import argparse
import sys
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights
import matplotlib.pyplot as plt

class V1_LinearLayer(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, center=None, scale=1, bias=False, seed=None):
        super(V1_LinearLayer, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=32) 
        self.clf = nn.Conv2d(in_channels=hidden_dim, out_channels=10, kernel_size=1)
        self.relu = nn.ReLU()
        
        # initialize the first layer
        V1_init(self.v1_layer, size, spatial_freq, center, scale, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
    def forward(self, x): #[128, 3, 32, 32]
        layer = self.v1_layer(x) #[128, hidden_dim, 1, 1]
        h = self.relu(layer)
        beta = self.clf(h) #[128, 10, 1, 1]
        return beta.squeeze() #[128, 10]
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) 
        loss.backward()
        optimizer.step()
        #if batch_idx % 50 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss.item()))
                
             
                      
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
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    """Train a simple Hybrid Resnet Scattering + CNN model on CIFAR.
        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.
        The model achieves around 88% testing accuracy after 10 epochs.
        scatter 1st order +
        scatter 2nd order + linear achieves 70.5% in 90 epochs
        scatter + cnn achieves 88% in 15 epochs
    """
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
    parser.add_argument('--width', type=int, default=2,help='width factor for resnet')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(32, 32), max_order=1)
        K = 17*3
    else:
        scattering = Scattering2D(J=2, shape=(32, 32))
        K = 81*3
    if use_cuda:
        scattering = scattering.cuda()



    h = 100
    
    s = 5
    f = 2
  
    scale = 1
    model = V1_LinearLayer(h, s, f, scale=scale, center=None).to(device)
 
    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
    lr = 0.000001 

    num_epoch = 90

    for epoch in range(0, num_epoch):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2

        train(model, device, train_loader, optimizer, epoch+1)
        test(model, device, test_loader, epoch+1)
    
    print("Hidden Dimension: ", h)