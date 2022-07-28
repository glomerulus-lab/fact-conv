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
sys.path.insert(0, '/home/whitev4/research/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights

class ScatterV1Linear(nn.Module):
    def __init__(self, in_channels, hidden_dim, size, spatial_freq, center=None, scale=1, bias=False, seed=None):
        super(ScatterV1Linear, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels, out_channels=hidden_dim, kernel_size=8) 
        self.clf = nn.Conv2d(in_channels=hidden_dim, out_channels=10, kernel_size=1)
        self.relu = nn.ReLU()
        self.K = in_channels
        self.bn = nn.BatchNorm2d(hidden_dim)
        
        # initialize the first layer
        V1_init(self.v1_layer, size, spatial_freq, center, scale, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
    def forward(self, x): #after scattering: [128, 3, 17, 8, 8]
        x = x.view(x.size(0), self.K, 8, 8) #[128, 51, 8, 8]
        layer = self.bn(self.v1_layer(x))
        h = self.relu(layer)
        beta = self.clf(h) #[128, 10, 1, 1]
        return beta.squeeze() #[128, 10]
    

def train(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data)) #[128, 10] and target is [128]
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % 50 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss.item()))

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
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
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



    h = 500
    s = 5
    f = 2
  
    scale = 1
    model = ScatterV1Linear(K, h, s, f, scale=scale, center=None).to(device)
    model.v1_layer.weight.requires_grad = True
 
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

    print("Hidden Dim: ", h)