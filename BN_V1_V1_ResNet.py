#Model w/ weights initialized according to V1 receptive fields & ResNet readout

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
import matplotlib.pyplot as plt
from datetime import datetime
import os
from distutils.util import strtobool
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
    
class BN_V1_V1_ResNet(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, k=2, n=4, num_classes=10, seed=None):
        super(BN_V1_V1_ResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(hidden_dim, eps=1e-5, affine=False),
            nn.Conv2d(hidden_dim, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )
        self.init_conv_x = nn.Sequential(
            nn.BatchNorm2d(3, eps=1e-5, affine=False),
            nn.Conv2d(3, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )
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
        
        scale1 = hidden_dim / ((3 * (32 * 32) ** 2) )
        scale2 = hidden_dim / ((hidden_dim * (32 * 32) ** 2))
        center = None
        
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
            
        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(1536, num_classes)
        
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
    def forward(self, x):  #[128, 3, 32, 32]
        x = self.relu(self.bn(x))
        h1 = self.relu(self.v1_layer(x))
        h2 = self.relu(self.v1_layer2(h1))
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)  
        x_pool = self.bn0(pool(x)) 
        h1_pool = self.bn1(pool(h1)) 
        h2_pool = self.bn2(pool(h2))
        
        x_res = self.init_conv_x(x_pool)
        h1_res = self.init_conv(h1_pool)
        h2_res = self.init_conv(h2_pool)
        
        x_res = self.layer2(x_res)
        h1_res = self.layer2(h1_res)
        h2_res = self.layer2(h2_res)
        
        x_res = self.layer3(x_res)
        h1_res = self.layer3(h1_res)
        h2_res = self.layer3(h2_res)
        
        x_res = self.avgpool(x_res)
        h1_res = self.avgpool(h1_res)
        h2_res = self.avgpool(h2_res)
        
        x_flat = x_res.view(x_res.size(0), -1)
        h1_flat = h1_res.view(h1_res.size(0), -1)
        h2_flat = h2_res.view(h2_res.size(0), -1)
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  #[128, 1536]
        
        beta = self.fc(concat) #[128, 10]
        return beta
    
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

def save_model(args, model):
    src = "/research/harris/vivian/v1-models/saved-models/BN_V1_V1_ResNet/"
    model_dir =  src + args.name
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = model_dir + "/trial_" + str(args.trial)
    if not os.path.exists(trial_dir): 
        os.makedirs(trial_dir)
    os.chdir(trial_dir)
    
    torch.save(test_loss, "loss.pt")
    torch.save(test_accuracy, "accuracy.pt")
    torch.save(model, "model.pt")
    torch.save(args, "args.pt")


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden dimensions in model')
    parser.add_argument('--num_epoch', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    parser.add_argument('--name', type=str, default='new model', help='filename for saved model')
    parser.add_argument('--trial', type=int, default=1, help='trial number')
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), default=False, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, help="which device to use (0 or 1)")
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    start = datetime.now()

    model = BN_V1_V1_ResNet(args.hidden_dim, args.s, args.f, args.scale, args.bias).to(device)

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
        ]), download=False),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    test_loss = []
    test_accuracy = []
    epoch_list = []

    for epoch in range(0, args.num_epoch):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005, nesterov=True)
            args.lr*=0.2

        train(model, device, train_loader, optimizer, epoch+1)
        loss, accuracy = test(model, device, test_loader, epoch+1)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
        epoch_list.append(epoch)

    end = datetime.now()
    print("Trial {} time (HH:MM:SS): {}".format(args.trial, end-start))
    print("Hidden dim: {}\t Learning rate: {}".format(args.hidden_dim, initial_lr))
    
    save_model(args, model)    
    
