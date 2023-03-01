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

device = "cuda:1" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, num_output_channels=3, filter_size=7): #does filter size change anything
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        # padding = (self.filter_size - 1) // 2
        padding = 3
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Regularized inverse scattering')
    parser.add_argument('--num_epochs', default=1, help='Number of epochs to train')
    parser.add_argument('--load_model', default=False, help='Load a trained model?')
    parser.add_argument('--dir_save_images', default='interpolation_images', help='Dir to save the sequence of images')
    parser.add_argument('--filename', default="V1 whitening", help='Dir to store model and results')
    parser.add_argument('--dim', default=100, help='num input channels')
    args = parser.parse_args()
    
    num_input_channels = int(args.dim)
    print("device: ", device)
    num_epochs = int(args.num_epochs)
    load_model = args.load_model
    dir_save_images = args.dir_save_images
    filename = "generative_scattering_results/"+args.filename
    print("filename: ", filename)

    dir_to_save = get_cache_dir(filename)

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,64)),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for reproducibility issues
    ])
    
    root = "/research/harris/vivian/v1-models/datasets/"
    train_dataset = datasets.CelebA(root=root, split="train", download=False, transform=transforms_to_apply)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=True) 

    fixed_dataloader_train = DataLoader(train_dataset, batch_size=2, shuffle=False)
    fixed_batch_train = next(iter(fixed_dataloader_train))
    fixed_batch_train = fixed_batch_train[0].float().to(device)

    scattering = V1_models.Generator_V1_celeba(num_input_channels, 128, 2, 0.1, 1, True).to(device)
    scattering.requires_grad = False
    scattering_fixed_batch_train = scattering(fixed_batch_train).squeeze(1) #2, 81, 7, 7
    
    test_dataset = datasets.CelebA(root=root, split="test", download=False, transform=transforms_to_apply)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=True) 

    fixed_dataloader_test = DataLoader(test_dataset, batch_size=2, shuffle=False)
    fixed_batch_test = next(iter(fixed_dataloader_test))
    fixed_batch_test = fixed_batch_test[0].float().to(device)
    scattering_fixed_batch_test = scattering(fixed_batch_test).squeeze(1)
    
    whitener = IncrementalPCA(n_components=num_input_channels, whiten=True)
    
    for idx_epoch in range(1): #2 epochs
        print('Whitening training epoch {}'.format(idx_epoch))
        for idx, batch in enumerate(train_dataloader): #469 batches
            images = batch[0].float().to(device)
            batch_scatter = scattering(images).view(images.size(0), -1).cpu().detach().numpy()
            whitener.partial_fit(batch_scatter)
    print("Done whitening")
    
    
    num_hidden_channels = 128

    generator = Generator(num_input_channels, num_hidden_channels).to(device)
    from torchsummary import summary
    #summary(generator, input_size=(3, 64, 64), device='cuda:1')
    
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
            for _, current_batch in enumerate(train_dataloader):
                generator.zero_grad()
                batch_images = Variable(current_batch[0]).float().to(device) #[128, 3, 128, 128]
                batch_scattering = scattering(batch_images).view(batch_images.size(0), -1).cpu().detach().numpy() #[128, 100]
                batch_whitened_scatter = torch.from_numpy(whitener.transform(batch_scattering)).float().to(device) #[128, 100]
                
                #breakpoint()
                batch_inverse_scattering = generator(batch_whitened_scatter)
                #print("made it here")
                loss = criterion(batch_inverse_scattering, batch_images)
                loss.backward()
                optimizer.step()
        print("Loss: ", loss)
        print('Saving results in {}'.format(dir_to_save))
       

        torch.save(generator.state_dict(), os.path.join(dir_to_save, 'model.pth'))

    generator.eval()

    
   
    # We create the batch containing the linear interpolation points in the scattering space
    ########################################################################################
   # fixed_dataloader = DataLoader(dataset, 2, shuffle=True)
    #fixed_batch = next(iter(fixed_dataloader))

   # z0 = fixed_batch['z'][[0]].numpy()
   # z1 = fixed_batch['z'][[1]].numpy()
#
   # batch_z = np.copy(z0)
        ####
    z0 = scattering_fixed_batch_train.cpu().detach().numpy()[[0]]
    z1 = scattering_fixed_batch_train.cpu().detach().numpy()[[1]]
    
    z2 = scattering_fixed_batch_test.cpu().detach().numpy()[[0]]
    z3 = scattering_fixed_batch_test.cpu().detach().numpy()[[1]]
    
    batch_z_train = np.copy(z0)
    batch_z_test = np.copy(z2)

    nb_samples = 32

    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:      
            zt = (1 - t) * z0 + t * z1
            batch_z_train = np.vstack((batch_z_train, zt))

    z = torch.from_numpy(batch_z_train).float().to(device)
    g_z = generator.forward(z)
    g_z = g_z.data.cpu().numpy().transpose((0, 2, 3, 1))


    for idx in range(nb_samples):
        filename_image = os.path.join(dir_to_save, '{}_train.png'.format(idx))
        Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)   
        
    for t in interval:
        if t > 0:      
            zt = (1 - t) * z2 + t * z3
            batch_z_test = np.vstack((batch_z_test, zt))

    z = torch.from_numpy(batch_z_test).float().to(device)
    g_z = generator.forward(z)
    g_z = g_z.data.cpu().numpy().transpose((0, 2, 3, 1))


    for idx in range(nb_samples):
        filename_image = os.path.join(dir_to_save, '{}_test.png'.format(idx))
        Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)   

#
    # code for generating 16 random images
    #####################################
    nb_samples = 16
    z = np.random.randn(nb_samples, num_input_channels)
    z = torch.from_numpy(z).float().to(device)
    g_z = generator.forward(z)
    filename_images = os.path.join(dir_to_save, 'train_random.png')
    temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    
    
    #code for reconstructing from train/test sets
    ####################################
    fixed_dataloader_train = DataLoader(train_dataset, 16)
    fixed_batch_train = next(iter(fixed_dataloader_train))
    fixed_batch_train = fixed_batch_train[0].float().to(device)
    
    scattering_fixed_batch_train = scattering(fixed_batch_train).squeeze(1)
    z = scattering_fixed_batch_train.cpu().detach().numpy()
    z = torch.from_numpy(z).float().to(device)
    
    g_z = generator.forward(z)
    filename_images = os.path.join(dir_to_save, 'train_reconstruct.png')
    temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    filename_images = os.path.join(dir_to_save, 'train_original.png')
    temp = make_grid(z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    
    #reconstructing test set
    fixed_dataloader_test = DataLoader(test_dataset, 16)
    fixed_batch_test = next(iter(fixed_dataloader_test))
    fixed_batch_test = fixed_batch_train[0].float().to(device)
    
    scattering_fixed_batch_test = scattering(fixed_batch_train).squeeze(1)
    z = scattering_fixed_batch_test.cpu().detach().numpy()
    z = torch.from_numpy(z).float().to(device)
  
    g_z = generator.forward(z)
    filename_images = os.path.join(dir_to_save, 'test_reconstruct.png')
    temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
    
    filename_images = os.path.join(dir_to_save, 'test_original.png')
    temp = make_grid(z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
    
   
