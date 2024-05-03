import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torch.nn.functional as F
import os
from torchvision.utils import make_grid

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)

class GSN(nn.Module):
    def __init__(self, nb_channels_input, z_dim):
        super(GSN, self).__init__()
        
        size_first_layer = 4
        self.main = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=size_first_layer * size_first_layer * nb_channels_input,
                      bias=True),
            View(-1, nb_channels_input, size_first_layer, size_first_layer),
            nn.BatchNorm2d(nb_channels_input, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),

            ConvBlock(nb_channels_input, int(nb_channels_input/2), upsampling=True),
            ConvBlock(int(nb_channels_input/2), int(nb_channels_input/4), upsampling=True),
            ConvBlock(int(nb_channels_input/4), int(nb_channels_input/8), upsampling=True),
            ConvBlock(int(nb_channels_input/8), int(nb_channels_input/16), upsampling=True),
            ConvBlock(int(nb_channels_input/16), nb_channels_output=3, tanh=True, upsampling=True)
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


class ConvBlock(nn.Module):
    def __init__(self, nb_channels_input, nb_channels_output, upsampling=False, tanh=False):
        super(ConvBlock, self).__init__()

        self.tanh = tanh
        self.upsampling = upsampling

        filter_size = 7
        padding = (filter_size - 1) // 2

        self.pad = nn.ReplicationPad2d(padding)
        self.conv = nn.Conv2d(nb_channels_input, nb_channels_output, filter_size, bias=True)
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


def interpolation_interval(whitened_batch, nb_samples):
    z0 = whitened_batch.cpu().detach().numpy()[[0]]
    z1 = whitened_batch.cpu().detach().numpy()[[1]]
    batch_z = np.copy(z0)

    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))
    return batch_z

def save_image(data, nrow, dir_to_save, imagename):
    filename = os.path.join(dir_to_save, imagename)
    temp = make_grid(data, nrow=nrow).cpu().numpy().transpose((1,2,0))
    Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename)

def save_interval(g_z, nb_samples, dir_to_save, train):
    for idx in range(nb_samples):
        filename = os.path.join(dir_to_save, train+'_{}.png'.format(idx))
        Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename)
