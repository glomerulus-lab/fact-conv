import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from kymatio.datasets import get_dataset_dir
import pdb
import torchvision
from sklearn.decomposition import IncrementalPCA
import V1_models

import V1_models_kam

from torchvision.utils import make_grid

from torch.utils.data import Subset
import torch.nn.functional as F
from pytorch_memlab import MemReporter
import generator_utils

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Regularized inverse scattering')
    parser.add_argument('--whiten_epochs', default=1, help='Number of epochs to train whitener')
    parser.add_argument('--train_epochs', default=1, help='Number of epochs to train generator')
    parser.add_argument('--load_model', default=False, help='Load a trained model?')
    parser.add_argument('--filename', default="V1_whitening", help='Dir to store model and results')
    parser.add_argument('--num_input_channels', default=1024, help='Number of input channels in generator')
    parser.add_argument('--hidden_dim', default=128, help='num hidden channels in V1 layers')
    parser.add_argument('--num_samples', default=16, help="num samples to interpolate")
    parser.add_argument('--batch_size', default=128, help="training batch size")
    parser.add_argument('--z_dim', default=512, help="whitening dimension")

    args = parser.parse_args()
    batch_size = int(args.batch_size)
    num_input_channels = int(args.num_input_channels)
    num_hidden_channels = int(args.hidden_dim)
    whiten_epochs = int(args.whiten_epochs)
    train_epochs = int(args.train_epochs)
    load_model = args.load_model
    nb_samples = int(args.num_samples)
    z_dim = int(args.z_dim)
    dir_to_save = "/research/harris/vivian/v1-models/generative_scattering_results/celeba/V1/"+args.filename
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    os.chdir(dir_to_save)

    print("device: ", device)

    transforms_to_apply = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for reproducibility issues
    ])
    
    root = "/research/harris/vivian/v1-models/datasets/"
    train_dataset = datasets.CelebA(root=root, split="train", download=False, transform=transforms_to_apply)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True) 
    whiten_dataloader = DataLoader(train_dataset, batch_size=z_dim, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
    test_dataset = datasets.CelebA(root=root, split="test", download=False, transform=transforms_to_apply)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    scattering = V1_models.Scattering_V1_celeba(num_input_channels, 2, 0.1, 1, True).to(device)
    scattering.requires_grad = False
    
    class Whitener(nn.Module):
        def __init__(self, z_dim, input_dim):
            super(Whitener, self).__init__()
            mean = torch.zeros(input_dim) 
            self.mean = nn.Parameter(mean)
            self.ll = nn.Linear(input_dim, z_dim, bias=False)

        def forward(self, x):
            xbar = x - self.mean
            z = self.ll(xbar)
            return z

    input_dim = 265216
    whitener = Whitener(z_dim, input_dim).to(device)
    opt = optim.Adam(whitener.parameters(), lr=0.00001) 
    cost_list = []
    mean_list = []
    linear_list = []
    #whitener = IncrementalPCA(n_components=z_dim, whiten=True)
    
    for idx_epoch in range(whiten_epochs): 
         print('Whitening training epoch {}'.format(idx_epoch))
         for idx, batch in enumerate(whiten_dataloader): #469 batches
             print('batch {}'.format(idx))
             whitener.zero_grad()
        
             images = batch[0].float().to(device)
             batch_scatter = scattering(images)
             batch_scatter = batch_scatter.view(images.size(0), -1)
             whitened = whitener(batch_scatter)
             mean_cost = torch.norm(whitener.mean - torch.mean(batch_scatter,\
                                                          axis=0)) ** 2
             linear_cost = torch.norm(torch.eye(whitened.shape[1],\
                                        whitened.shape[1]).to(device)\
                                      - torch.mm(whitened.T, whitened)\
                                                   / batch_size) ** 2
             cost = mean_cost + linear_cost
             cost_list.append(cost)
             mean_list.append(mean_cost)
             linear_list.append(linear_cost)
             print("whitening cost: ", cost)
             cost.backward()
             opt.step()
            
             
    print("Done whitening")
    torch.save(torch.Tensor(cost_list), os.path.join(dir_to_save,\
            'whitening_cost.pt')) 
    torch.save(torch.Tensor(linear_list), os.path.join(dir_to_save,\
            'linear_cost.pt'))
    torch.save(torch.Tensor(mean_list), os.path.join(dir_to_save,\
            'mean_cost.pt'))
    generator = generator_utils.GSN(nb_channels_input=num_input_channels, z_dim=z_dim).to(device)    
    generator.train()
    
    # Either train the network or load a trained model
    if load_model:
        filename_model = os.path.join(dir_to_save, 'model.pth')
        generator.load_state_dict(torch.load(filename_model))
    else:
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(generator.parameters(), lr=0.0005)
        
        for idx_epoch in range(train_epochs):
            print('Generator training epoch {}'.format(idx_epoch))
            for _, current_batch in enumerate(train_dataloader):
                generator.zero_grad()
                batch_images = Variable(current_batch[0]).float().to(device) 
                batch_scattering = scattering(batch_images).view(batch_images.size(0), -1).cpu().detach().numpy() 
                batch_whitened_scatter = torch.from_numpy(whitener.transform(batch_scattering)).float().to(device) 
                batch_inverse_scattering = generator(batch_whitened_scatter)
                loss = criterion(batch_inverse_scattering, batch_images)
                loss.backward()
                optimizer.step()
            print("Epoch: {} Loss: {}".format(idx_epoch, loss))
        torch.save(generator.state_dict(), os.path.join(dir_to_save, 'model.pth'))
    generator.eval()
    
    print('Saving results in {}'.format(dir_to_save))

    # Linear Interpolation from train set
    fixed_dataloader_train = DataLoader(train_dataset, batch_size=2, shuffle=False)
    fixed_batch_train = next(iter(fixed_dataloader_train))
    fixed_batch_train = fixed_batch_train[0].float().to(device)
    scattering_fixed_batch_train = scattering(fixed_batch_train).squeeze(1).cpu().detach().numpy()
    whitened_batch_train = torch.from_numpy(whitener.transform(scattering_fixed_batch_train)).float().to(device)
    batch_z_train = generator_utils.interpolation_interval(whitened_batch_train, nb_samples)
    z = torch.from_numpy(batch_z_train).float().to(device) 
    gz = generator.forward(z)
    g_z = gz.data.cpu().numpy().transpose((0, 2, 3, 1))
    generator_utils.save_interval(g_z, nb_samples, dir_to_save, 'train')
    generator_utils.save_image(gz.data[:16], nb_samples, dir_to_save, 'train_interpolation.png')
    
    #print("INTERPOLATION MIN VALUE: ", torch.min(gz))
    #print("INTERPOLATION MAX VALUE: ", torch.max(gz))

    #print("RGB MIN VALUE: ", torch.min(gz, dim=1))
    #print("RGB MAX VALUE: ", torch.max(gz, dim=1))

    # Linear Interpolation from test set
    fixed_dataloader_test = DataLoader(test_dataset, batch_size=2, shuffle=False)
    fixed_batch_test = next(iter(fixed_dataloader_test))
    fixed_batch_test = fixed_batch_test[0].float().to(device)
    scattering_fixed_batch_test = scattering(fixed_batch_test).squeeze(1).cpu().detach().numpy()
    whitened_batch_test = torch.from_numpy(whitener.transform(scattering_fixed_batch_test)).float().to(device)
    batch_z_test = generator_utils.interpolation_interval(whitened_batch_test, nb_samples)
    z = torch.from_numpy(batch_z_test).float().to(device)
    gz = generator.forward(z)
    g_z = gz.data.cpu().numpy().transpose((0, 2, 3, 1))
    generator_utils.save_interval(g_z, nb_samples, dir_to_save, 'test')
    generator_utils.save_image(gz.data[:nb_samples], nb_samples, dir_to_save, 'test_interpolation.png')
    
    # Generating 16 random images
    nb_samples = 16
    z = np.random.randn(nb_samples, z_dim)
    z = torch.from_numpy(z).float().to(device)
    g_z = generator.forward(z)
    generator_utils.save_image(g_z.data[:nb_samples], 4, dir_to_save, 'random_generation.png')
    
    # Reconstructing from train set
    fixed_dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=False)
    fixed_batch_train = next(iter(fixed_dataloader_train))
    fixed_batch_train = fixed_batch_train[0].float().to(device) 
    scattering_fixed_batch_train = scattering(fixed_batch_train).squeeze(1) 
    ztrain = scattering_fixed_batch_train.cpu().detach().numpy()
    whiten_z_train = torch.from_numpy(whitener.transform(ztrain)).float().to(device)
    g_ztrain = generator.forward(whiten_z_train)
    generator_utils.save_image(g_ztrain.data[:16], 4, dir_to_save, 'train_reconstruct.png')
    generator_utils.save_image(fixed_batch_train, 4, dir_to_save, 'train_original.png')
    
    # Reconstructing from test set
    fixed_dataloader_test = DataLoader(test_dataset, batch_size=16, shuffle=False)
    fixed_batch_test = next(iter(fixed_dataloader_test))
    fixed_batch_test = fixed_batch_test[0].float().to(device) 
    scattering_fixed_batch_test = scattering(fixed_batch_test).squeeze(1)
    ztest = scattering_fixed_batch_test.cpu().detach().numpy()
    whiten_z_test = torch.from_numpy(whitener.transform(ztest)).float().to(device)
    g_ztest = generator.forward(whiten_z_test)  
    generator_utils.save_image(g_ztest.data[:16], 4, dir_to_save, 'test_reconstruct.png')
    generator_utils.save_image(fixed_batch_test, 4, dir_to_save, 'test_original.png')
