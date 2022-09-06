#Control model w/ weights initialized from a Gaussian process with diagonal covariance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
from distutils.util import strtobool
import argparse
import sys
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights
import matplotlib.pyplot as plt
from datetime import datetime
import os

class Control_Uniform(nn.Module):
    def __init__(self, hidden_dim, bias, seed=None):
        super(Control_Uniform, self).__init__()
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
        
    def forward(self, x):  #[128, 3, 32, 32]
        h1 = self.relu(self.v1_layer(self.bn(x)))  #[128, hidden_dim, 32, 32] w/ k=7, s=1, p=3
        h2 = self.relu(self.v1_layer2(h1))  #[128, hidden_dim, 32, 32] w/ k=7, s=1, p=3
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)  
        x_pool = self.bn0(pool(x))  #[128, 3, 8, 8]
        h1_pool = self.bn1(pool(h1))  #[128, hidden_dim, 8, 8]
        h2_pool = self.bn2(pool(h2))  #[128, hidden_dim, 8, 8]
        
        x_flat = x_pool.view(x_pool.size(0), -1)  #[128, 192] = [128, 3 * 8 * 8] std ~1, mean ~0
        h1_flat = h1_pool.view(h1_pool.size(0),  -1)  #[128, hidden_dim * 8 * 8] std ~1, mean ~0
        h2_flat = h2_pool.view(h2_pool.size(0), -1)  #[128, hidden_dim * 8 * 8] std ~1, mean ~0
        
        concat = torch.cat((x_flat, h1_flat, h2_flat), 1)  #[128, (3 * 8 * 8) + (hidden_dim * 8 * 8) + (hidden_dim * 8 * 8)
        
        beta = self.clf(concat) #[128, 10]
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

def save_model(name, trial, model):
    src = "/research/harris/vivian/v1-models/saved-models/Control_Uniform_CIFAR100/"
    model_dir =  src + name
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    os.chdir(model_dir)
    
            
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = model_dir + "/trial_" + str(trial)
    if not os.path.exists(trial_dir): 
        os.makedirs(trial_dir)
    os.chdir(trial_dir)
    
    torch.save(test_loss, "loss.pt")
    torch.save(test_accuracy, "accuracy.pt")
    torch.save(model, "model.pt")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden dimensions in model')
    parser.add_argument('--num_epoch', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--name', type=str, default="new_model", help="file name")
    parser.add_argument('--trial', type=int, default=1, help="trial #")
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), default=False, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, help="which device to use (0 or 1)")
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")


    model = Control_Uniform(args.hidden_dim, args.bias).to(device)
 
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
        datasets.CIFAR100(root=scattering_datasets.get_dataset_dir('CIFAR100'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=scattering_datasets.get_dataset_dir('CIFAR100'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
 
    test_loss = []
    test_accuracy = []
    epoch_list = []
    
    start = datetime.now()

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
    
    save_model(args.name, args.trial, model)
