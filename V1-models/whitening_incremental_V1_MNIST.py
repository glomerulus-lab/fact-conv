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

import pdb
import torchvision
from sklearn.decomposition import IncrementalPCA
import V1_models

from torchvision.utils import make_grid

device = "cuda:1" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_output_channels=1, filter_size=3):
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2

        self.main = nn.Sequential(
            nn.Linear(self.num_input_channels, self.num_hidden_channels * 7 * 7),
            nn.Unflatten(1, (self.num_hidden_channels, 7, 7)),
            nn.ReLU(inplace=True),
       
            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Regularized inverse scattering')
    parser.add_argument('--num_epochs', default=1, help='Number of epochs to train')
    parser.add_argument('--load_model', default=False, help='Load a trained model?')
    parser.add_argument('--dir_save_images', default='interpolation_images', help='Dir to save the sequence of images')
    parser.add_argument('--filename', default="V1 whitening", help='Dir to store model and results')
    parser.add_argument('--dim', default=100, help='num input channels')
    args = parser.parse_args()
    
    num_input_channels = int(args.dim)
    
    num_epochs = int(args.num_epochs)
    load_model = args.load_model
    dir_save_images = args.dir_save_images
    filename = "generative_scattering_results/"+args.filename
    print("filename: ", filename)

    dir_to_save = get_cache_dir(filename)

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for reproducibility issues
    ])
    
    mnist_dir = get_dataset_dir("MNIST", create=True)
    dataset = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms_to_apply)
    
        
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=True) 

    fixed_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    fixed_batch = next(iter(fixed_dataloader))
    fixed_batch = fixed_batch[0].float().to(device)

    scattering = V1_models.Scattering_V1_MNIST(num_input_channels, 2, 0.1, 1, True).to(device)
    scattering.requires_grad = False
    scattering_fixed_batch = scattering(fixed_batch).squeeze(1) #2, 81, 7, 7
    
    whitener = IncrementalPCA(n_components=num_input_channels, whiten=True)
 
    for idx_epoch in range(1): #2 epochs
        print('Whitening training epoch {}'.format(idx_epoch))
        for idx, batch in enumerate(dataloader): #469 batches
            images = batch[0].float().to(device)
            batch_scatter = scattering(images).view(images.size(0), -1).cpu().detach().numpy()
            
            whitener.partial_fit(batch_scatter)
        
    
    num_hidden_channels = 256

    generator = Generator(num_input_channels, num_hidden_channels).to(device)
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
            print('Generator training epoch {}'.format(idx_epoch))
            for _, current_batch in enumerate(dataloader):
                generator.zero_grad()
                batch_images = Variable(current_batch[0]).float().to(device)
                batch_scattering = scattering(batch_images).view(batch_images.size(0), -1).cpu().detach().numpy()
                batch_whitened_scattering = torch.from_numpy(whitener.transform(batch_scattering)).float().to(device)
                batch_inverse_scattering = generator(batch_whitened_scattering)
                loss = criterion(batch_inverse_scattering, batch_images)
                loss.backward()
                optimizer.step()
        print("Loss: ", loss)
        print('Saving results in {}'.format(dir_to_save))
       

        torch.save(generator.state_dict(), os.path.join(dir_to_save, 'model.pth'))

    generator.eval()

   
    # We create the batch containing the linear interpolation points in the scattering space
    ########################################################################################
    z0 = scattering_fixed_batch.cpu().detach().numpy()[[0]]
    z1 = scattering_fixed_batch.cpu().detach().numpy()[[1]]
    batch_z = np.copy(z0)
    num_samples = 32
    interval = np.linspace(0, 1, num_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    z = torch.from_numpy(batch_z).float().to(device)
    z_view = z.view(z.size(0), -1).cpu().numpy()
    z_whitened = torch.from_numpy(whitener.transform(z_view)).float().to(device)
   
    path = generator(z_whitened).data.cpu().numpy().squeeze(1)
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
        
    # code for generating 16 random images
    #####################################
    nb_samples = 16
    z = np.random.randn(nb_samples, num_input_channels)
    z = torch.from_numpy(z).float().to(device)
    g_z = generator.forward(z)
    filename_images = os.path.join(dir_to_save, 'epoch_random.png')
    temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    
   
