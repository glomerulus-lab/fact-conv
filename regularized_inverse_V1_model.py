"""
Regularized inverse of a scattering transform on MNIST
======================================================

Description:
This example trains a convolutional network to invert the scattering transform at scale 2 of MNIST digits.
After only two epochs, it produces a network that transforms a linear interpolation in the scattering space into a
nonlinear interpolation in the image space.

Remarks:
The model after two epochs and the path (which consists of a sequence of images) are stored in the cache directory.
The two epochs take roughly 5 minutes in a Quadro M6000.

Reference:
https://arxiv.org/abs/1805.06621
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from kymatio.torch import Scattering2D as Scattering
from kymatio.caching import get_cache_dir
from kymatio.datasets import get_dataset_dir

from distutils.util import strtobool
import sys
sys.path.insert(0, '/research/harris/vivian/structured_random_features/')
from src.models.init_weights import V1_init, classical_init, V1_weights

from torchvision import models
from torchsummary import summary

import BN_V1_V1_Linear_MNIST



device = "cuda" if torch.cuda.is_available() else "cpu"

class BN_V1_V1(nn.Module):
    def __init__(self, hidden_dim, size, spatial_freq, scale, bias, seed=None):
        super(BN_V1_V1_LinearLayer, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                  bias=bias) 
        self.v1_layer2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, 
                                   bias=bias)
        #self.clf = nn.Linear((1 * (8 ** 2)) + (hidden_dim * (8 ** 2)) + (hidden_dim * (8 ** 2)), 100)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        scale1 = hidden_dim / ((1 * (28 * 28) ** 2) )
        scale2 = hidden_dim / ((hidden_dim * (28 * 28) ** 2))
        center = None
        
        V1_init(self.v1_layer, size, spatial_freq, center, scale1, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size, spatial_freq, center, scale2, bias, seed)
        self.v1_layer2.weight.requires_grad = False
        
        if bias==True:
            self.v1_layer.bias.requires_grad = False
            self.v1_layer2.bias.requires_grad = False
        
    def forward(self, x):  #[128, 1, 28, 28]
        h1 = self.relu(self.v1_layer(self.bn(x)))  #[128, hidden_dim, 28, 28] w/ k=7, s=1, p=3
        h2 = self.relu(self.v1_layer2(h1))  #[128, hidden_dim, 28, 28] w/ k=7, s=1, p=3
        
        pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)  
        x_pool = self.bn0(pool(x))  #[128, 1, 8, 8]
        h1_pool = self.bn1(pool(h1))  #[128, hidden_dim, 8, 8]
        h2_pool = self.bn2(pool(h2))  #[128, hidden_dim, 8, 8]
        
        concat = torch.cat((x, h1, h2), 1)  #[128, (3 * 8 * 8) + (hidden_dim * 8 * 8) + (hidden_dim * 8 * 8)
        
        #beta = self.clf(concat) #[128, 10]
        return concat

class Generator(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, bias, num_output_channels=1, filter_size=3):
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        
        self.v1_layer1 = nn.Conv2d(num_input_channels, num_hidden_channels, filter_size, bias=False)
        self.v1_layer2 = nn.Conv2d(num_hidden_channels, num_hidden_channels, filter_size, bias=False)
        
        scale1 = num_hidden_channels / ((1 * (28 * 28) ** 2) )
        scale2 = num_hidden_channels / ((num_hidden_channels * (28 * 28) ** 2))
        center = None
        
        V1_init(self.v1_layer1, size=2, spatial_freq=0.1, center=center, scale=scale1, bias=bias)
        self.v1_layer1.weight.requires_grad = False
        
        V1_init(self.v1_layer2, size=2, spatial_freq=0.1, center=center, scale=scale2, bias=bias)
        self.v1_layer2.weight.requires_grad = False
        
        self.build()
        

    def build(self):
        padding = (self.filter_size - 1) // 2

        self.main = nn.Sequential(
            nn.ReflectionPad2d(padding),
            self.v1_layer1,
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            self.v1_layer2,
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9),
            nn.Tanh())

    def forward(self, input_tensor):
        return self.main(input_tensor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Regularized inverse scattering')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--load_model', default=False, help='Load a trained model?')
    parser.add_argument('--dir_save_images', default='interpolation_images', help='Dir to save the sequence of images')
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), default=False, help='bias=True or False')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden dimensions in model')
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency') 
    parser.add_argument('--name', type=str, default='reg_inverse_example', help='cache directory name')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    print("Num epochs: ", num_epochs)
    load_model = args.load_model
    
    dir_save_images = args.dir_save_images
    
    src = "/research/harris/vivian/v1-models/saved-models/regularized_inverse_scattering/"
    model_dir =  src + args.name
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    os.chdir(model_dir)
    
   
    dir_to_save = model_dir

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for reproducibility issues
    ])

    mnist_dir = get_dataset_dir("MNIST", create=True)
    dataset = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms_to_apply)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)

    fixed_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    fixed_batch = next(iter(fixed_dataloader))
    fixed_batch = fixed_batch[0].float().to(device)

    #scattering = Scattering(J=2, shape=(28, 28)).to(device)
    scattering = BN_V1_V1().to(device)

    scattering_fixed_batch = scattering(fixed_batch).squeeze(1)
    num_input_channels = scattering_fixed_batch.shape[1]
    num_hidden_channels = args.hidden_dim

    generator = Generator(num_input_channels, num_hidden_channels, args.bias).to(device)
    
    #summary(generator, (81, 7, 7), batch_size=128, device='cuda')
    
    generator.train()

    # Either train the network or load a trained model
    ##################################################
    if load_model:
        filename_model = os.path.join(dir_to_save, 'model.pth')
        generator.load_state_dict(torch.load(filename_model))
    else:
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(generator.parameters())

        for idx_epoch in range(num_epochs):
            print('Training epoch {}'.format(idx_epoch))
            for _, current_batch in enumerate(dataloader):
                generator.zero_grad()
                batch_images = Variable(current_batch[0]).float().to(device)
                batch_scattering = scattering(batch_images).squeeze(1)
                batch_inverse_scattering = generator(batch_scattering)
                loss = criterion(batch_inverse_scattering, batch_images)
                loss.backward()
                optimizer.step()

        print('Saving results in {}'.format(dir_to_save))

        torch.save(generator.state_dict(), os.path.join(dir_to_save, 'model.pth'))

    generator.eval()

    # We create the batch containing the linear interpolation points in the scattering space
    ########################################################################################
    z0 = scattering_fixed_batch.cpu().numpy()[[0]]
    z1 = scattering_fixed_batch.cpu().numpy()[[1]]
    batch_z = np.copy(z0)
    num_samples = 32
    interval = np.linspace(0, 1, num_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    z = torch.from_numpy(batch_z).float().to(device)
    path = generator(z).data.cpu().numpy().squeeze(1)
    path = (path + 1) / 2  # The pixels are now in [0, 1]

    # We show and store the nonlinear interpolation in the image space
    ##################################################################
    dir_path = os.path.join(dir_to_save, dir_save_images)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for idx_image in range(num_samples):
        current_image = np.uint8(path[idx_image] * 255.0)
        filename = os.path.join(dir_path, '{}.png'.format(idx_image))
        Image.fromarray(current_image).save(filename)
        plt.imshow(current_image, cmap='gray')
        plt.axis('off')
        plt.pause(0.1)
        plt.draw()
