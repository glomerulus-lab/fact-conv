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

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class V1_ResNet(nn.Module):
    def __init__(self, in_channels,  hidden_dim, size, spatial_freq, center=None, scale=1, bias=False, seed=None, k=2, n=4, num_classes=10):
        super(V1_ResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(hidden_dim, eps=1e-5, affine=False),
            nn.Conv2d(hidden_dim, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )
        self.v1_layer = nn.Conv2d(3, out_channels=hidden_dim, kernel_size=32, stride=1) 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(hidden_dim)
       
        # initialize the first layer
        V1_init(self.v1_layer, size, spatial_freq, center, scale, bias, seed)
        self.v1_layer.weight.requires_grad = False

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): #[128, 3, 32, 32]
        v1 = self.bn(self.v1_layer(x)) #[128, hidden_dim, 4, 4]
        h = self.relu(v1) 
        x = self.init_conv(h) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) #[128, 10]
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        train_loss = loss.item()
        
        #batch_percent = round(100. * batch_idx / len(train_loader))
        #if batch_percent == 100:
            #print('Train Epoch: {}, Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                #epoch, train_loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))

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
    print('Test Epoch: {}, Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


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



    h = 500 #50 100 400 1000 2000 5000
    s = 5
    f = 2
  
    scale = 1
    model = V1_ResNet(K, h, s, f, scale=scale, center=None).to(device)
 
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

        train(model, device, train_loader, optimizer, epoch+1)
        test(model, device, test_loader, epoch+1)

