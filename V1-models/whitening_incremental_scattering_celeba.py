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
from torch.utils.data import Subset

import torch.nn.functional as F

device = "cuda:1" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_output_channels=3, filter_size=5): #does filter size change anything
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2
        
        self.main = nn.Sequential(
            nn.Linear(self.num_input_channels, self.num_hidden_channels * 4 * 4),
            nn.Unflatten(1, (self.num_hidden_channels, 4, 4)), #changing these - image dim / 4 -- WHY
            nn.ReLU(inplace=True),
       
            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # added
            nn.ReflectionPad2d(padding),
            nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
            nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # added
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
        #print("pre main shape: ", input_tensor.size())
        #print("post main shape: ", self.main(input_tensor).size())
        return self.main(input_tensor)

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)
    
class NewGenerator(nn.Module): # SCATTERING GENERATOR!!!
    def __init__(self, nb_channels_first_layer=4, z_dim=2048, size_first_layer=4):
        super(NewGenerator, self).__init__()

        #nb_channels_input = nb_channels_first_layer * 32  #32 --> 128; 16 --> 64
        nb_channels_input = nb_channels_first_layer * 32
        
        self.main = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=size_first_layer * size_first_layer * nb_channels_input,
                      bias=False),
            View(-1, nb_channels_input, size_first_layer, size_first_layer),
            nn.BatchNorm2d(nb_channels_input, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),

            ConvBlock(nb_channels_first_layer * 32, nb_channels_first_layer * 16, upsampling=True),
            ConvBlock(nb_channels_first_layer * 16, nb_channels_first_layer * 8, upsampling=True),
            ConvBlock(nb_channels_first_layer * 8, nb_channels_first_layer * 4, upsampling=True),
            ConvBlock(nb_channels_first_layer * 4, nb_channels_first_layer * 2, upsampling=True),
            ConvBlock(nb_channels_first_layer * 2, nb_channels_first_layer, upsampling=True),

            ConvBlock(nb_channels_first_layer, nb_channels_output=3, tanh=True) #goal is shape 128x128
        )

    def forward(self, input_tensor):
        #print("Input shape: ", input_tensor.shape)
        #print("Output shape: ", self.main(input_tensor).shape)
        return self.main(input_tensor)


class ConvBlock(nn.Module):
    def __init__(self, nb_channels_input, nb_channels_output, upsampling=False, tanh=False):
        super(ConvBlock, self).__init__()

        self.tanh = tanh
        self.upsampling = upsampling

        filter_size = 5
        padding = (filter_size - 1) // 2

        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(nb_channels_input, nb_channels_output, filter_size, bias=False)
        self.bn_layer = nn.BatchNorm2d(nb_channels_output, eps=0.001, momentum=0.9)

    def forward(self, input_tensor):
        if self.upsampling:
            output = F.interpolate(input_tensor, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            output = input_tensor

        output = self.pad(output)
        output = self.conv(output)
        output = self.bn_layer(output)

        if self.tanh:
            output = torch.tanh(output)
        else:
            output = F.relu(output)

        return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Regularized inverse scattering')
    parser.add_argument('--whiten_epochs', default=1, help='Number of epochs to whiten')
    parser.add_argument('--train_epochs', default=1, help='Number of epochs to train')
    parser.add_argument('--load_model', default=False, help='Load a trained model?')
    parser.add_argument('--dir_save_images', default='interpolation_images', help='Dir to save the sequence of images')
    parser.add_argument('--filename', default="V1_whitening", help='Dir to store model and results')
    parser.add_argument('--dim', default=100, help='num input channels')
    parser.add_argument('--num_samples', default=32, help="num samples for image interpolation")
    args = parser.parse_args()
    
    num_input_channels = int(args.dim)
    print("cuda available? ", torch.cuda.is_available())
    print("device: ", device)
    train_epochs = int(args.train_epochs)
    whiten_epochs = int(args.whiten_epochs)
    load_model = args.load_model
    dir_save_images = args.dir_save_images
    nb_samples = int(args.num_samples)
    filename = "generative_scattering_results/celeba/scattering/"+args.filename
    print("filename: ", filename)

    dir_to_save = get_cache_dir(filename)

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for reproducibility issues
    ])
    
    root = "/research/harris/vivian/v1-models/datasets/"
    train_dataset = datasets.CelebA(root=root, split="train", download=False, transform=transforms_to_apply)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=True) 
    test_dataset = datasets.CelebA(root=root, split="test", download=False, transform=transforms_to_apply)

    scattering = Scattering(J=2, shape=(128,128)).to(device)
    
    whitener = IncrementalPCA(n_components=num_input_channels, whiten=True)
    # whitener = IncrementalPCA(n_components=32, whiten=True)

    for idx_epoch in range(whiten_epochs): #2 epochs
        print('Whitening training epoch {}'.format(idx_epoch))
        for idx, batch in enumerate(train_dataloader): #469 batches
            images = batch[0].float().to(device)
            batch_scatter = scattering(images).view(images.size(0), -1).cpu().detach().numpy()
            whitener.partial_fit(batch_scatter)
            
    print("Done whitening")
    
    
    num_hidden_channels = 128

    #generator = Generator(num_input_channels, num_hidden_channels).to(device)
    print("Num input channels: ", num_input_channels)
    print("Num hidden channels: ", num_hidden_channels)
    generator = NewGenerator(nb_channels_first_layer=4, z_dim=128, size_first_layer=4).to(device)
        
    generator.train()
    
    # Either train the network or load a trained model
    ##################################################
    
    if load_model:
        filename_model = os.path.join(dir_to_save, 'model.pth')
        generator.load_state_dict(torch.load(filename_model))
    else:
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(generator.parameters())

        for idx_epoch in range(train_epochs):
            print('Generator training epoch {}'.format(idx_epoch))
            for _, current_batch in enumerate(train_dataloader):
                generator.zero_grad()
                batch_images = Variable(current_batch[0]).float().to(device) #[128, 3, 128, 128]
                batch_scattering = scattering(batch_images).view(batch_images.size(0), -1).cpu().detach().numpy() #[128, 248832]
                batch_whitened_scatter = torch.from_numpy(whitener.transform(batch_scattering)).float().to(device) #[128, 128]
                batch_inverse_scattering = generator(batch_whitened_scatter) 
                loss = criterion(batch_inverse_scattering, batch_images) 
                loss.backward()
                optimizer.step()

        print("Loss: ", loss)
        print('Saving results in {}'.format(dir_to_save))
       

        torch.save(generator.state_dict(), os.path.join(dir_to_save, 'model.pth'))

    generator.eval()

    
   
    # We create the batch containing the linear interpolation points in the scattering space
    ########################################################################################
    
    # whiten & scatter training data
    fixed_dataloader_train = DataLoader(train_dataset, batch_size=2, shuffle=False)
    fixed_batch_train = next(iter(fixed_dataloader_train))
    fixed_batch_train = fixed_batch_train[0].float().to(device) #2, 3, 128, 128
    scattering_fixed_batch_train = scattering(fixed_batch_train).squeeze(1) #[2, 3, 81, 32, 32]
    scattering_fixed_batch_train = scattering_fixed_batch_train.reshape([2, -1]).cpu().detach().numpy()
    whitened_batch_train = torch.from_numpy(whitener.transform(scattering_fixed_batch_train)).float().to(device)
    z0 = whitened_batch_train.cpu().detach().numpy()[[0]] 
    z1 = whitened_batch_train.cpu().detach().numpy()[[1]]
    batch_z_train = np.copy(z0)

    # whiten & scatter testing data
    fixed_dataloader_test = DataLoader(test_dataset, batch_size=2, shuffle=False)
    fixed_batch_test = next(iter(fixed_dataloader_test))
    fixed_batch_test = fixed_batch_test[0].float().to(device)
    scattering_fixed_batch_test = scattering(fixed_batch_test).squeeze(1)
    scattering_fixed_batch_test = scattering_fixed_batch_test.reshape([2, -1]).cpu().detach().numpy() 
    whitened_batch_test = torch.from_numpy(whitener.transform(scattering_fixed_batch_test)).float().to(device)
    z2 = whitened_batch_test.cpu().detach().numpy()[[0]]
    z3 = whitened_batch_test.cpu().detach().numpy()[[1]]
    batch_z_test = np.copy(z2)

    # set up training path
    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:      
            zt = (1 - t) * z0 + t * z1
            batch_z_train = np.vstack((batch_z_train, zt))
    
    # run training path thru generator
    z = torch.from_numpy(batch_z_train).float().to(device) 
    gz = generator.forward(z) 
    g_z = gz.data.cpu().numpy().transpose((0, 2, 3, 1))

    # save individual training images
    for idx in range(nb_samples):
        filename_image = os.path.join(dir_to_save, '{}_train.png'.format(idx))
        Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)   

    # save single training image
    filename_images = os.path.join(dir_to_save, 'interpolation_train.png')
    temp = make_grid(gz.data[:nb_samples], nrow=16).cpu().numpy().transpose((1,2,0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)

    # set up testing path
    for t in interval:
        if t > 0:      
            zt = (1 - t) * z2 + t * z3
            batch_z_test = np.vstack((batch_z_test, zt))

    # run testing path thru generator
    z = torch.from_numpy(batch_z_test).float().to(device)
    gz = generator.forward(z)
    g_z = gz.data.cpu().numpy().transpose((0, 2, 3, 1))

    # save individual testing images
    for idx in range(nb_samples):
        filename_image = os.path.join(dir_to_save, '{}_test.png'.format(idx))
        Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)   

    # save single testing image
    filename_images = os.path.join(dir_to_save, 'interpolation_test.png')
    temp = make_grid(gz.data[:nb_samples], nrow=16).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)


    # code for generating 16 random images
    #####################################
    
    z = np.random.randn(nb_samples, num_input_channels)
    z = torch.from_numpy(z).float().to(device)
    g_z = generator.forward(z)

    filename_images = os.path.join(dir_to_save, 'train_random.png')
    temp = make_grid(gz.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    
    
    #code for reconstructing from train/test sets
    #############################################
    
    # scatter & whiten training data
    fixed_dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=False)
    fixed_batch_train = next(iter(fixed_dataloader_train))
    fixed_batch_train = fixed_batch_train[0].float().to(device)
    scattering_fixed_batch_train = scattering(fixed_batch_train).squeeze(1).reshape([16, -1])
    ztrain = scattering_fixed_batch_train.cpu().detach().numpy()
    whiten_z_train = torch.from_numpy(whitener.transform(ztrain)).float().to(device)

    # save original training image
    filename_images = os.path.join(dir_to_save, 'train_original.png')
    temp = make_grid(fixed_batch_train, nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
        
    # scatter & whiten testing data
    fixed_dataloader_test = DataLoader(test_dataset, batch_size=16, shuffle=False)
    fixed_batch_test = next(iter(fixed_dataloader_test))
    fixed_batch_test = fixed_batch_test[0].float().to(device)
    scattering_fixed_batch_test = scattering(fixed_batch_test).squeeze(1).reshape([16, -1])
    ztest = scattering_fixed_batch_test.cpu().detach().numpy()
    whiten_z_test = torch.from_numpy(whitener.transform(ztest)).float().to(device)

    # save original testing image
    filename_images = os.path.join(dir_to_save, 'test_original.png')
    temp = make_grid(fixed_batch_test, nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    # save reconstructed training images
    g_z1 = generator.forward(whiten_z_train)
    filename_images = os.path.join(dir_to_save, 'train_reconstruct.png')
    temp = make_grid(g_z1.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    # save reconstructed testing images
    g_z2 = generator.forward(whiten_z_test)
    filename_images = os.path.join(dir_to_save, 'test_reconstruct.png')
    temp = make_grid(g_z2.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
   
