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

class ScatteringLinearLayer(nn.Module):
    def __init__(self, in_channels, num_classes = 10):
        super(ScatteringLinearLayer, self).__init__()
        self.clf = nn.Linear(1004, 10)
        hidden_dim=100
        self.v1_layer = nn.Conv2d(in_channels=51, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=False) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(51)
        self.bn0 = nn.BatchNorm2d(51)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        
        
        self.v1_layer.weight.requires_grad = False
        self.v1_layer2.weight.requires_grad = False
       
    def forward(self, x): #original shape: (128, 3, 17, 8, 8)
        x = x.view(x.size(0), 51, 8, 8)
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
        
        beta = self.clf(concat) #[128, 10]
        return beta



def train(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, scattering):
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


    model = ScatteringLinearLayer(args.width).to(device)

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
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
    lr = 0.1
    for epoch in range(0, 90):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2

        train(model, device, train_loader, optimizer, epoch+1, scattering)
        test(model, device, test_loader, scattering)
