'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import math
from conv_modules import NewResamplingDoubleFactConv2d



class Concat(nn.Module):
    def forward(self, x):
        return torch.cat([x, x], dim=1)


def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + 1e-4)

class SVD(torch.autograd.Function):
    """Low-rank SVD with manually re-implemented gradient.

    This function calculates the low-rank SVD decomposition of an arbitary
    matrix and re-implements the gradient such that we can regularize the
    gradient.

    Parameters
    ----------
    A : tensor
        Input tensor with at most 3 dimensions. Usually is 2 dimensional. if
        3 dimensional the svd is batched over the first dimension.

    size : int
        Slightly over-estimated rank of A.
    """
    @staticmethod
    def forward(self, A, size):
        U, S, V = torch.svd_lowrank(A, size, 2)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G 

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA, None



class Alignment(nn.Module):
    """Procurstes Alignment module.

    This module splits the input along the channel/2nd dimension into the generated
    path and reference path, calculates the cross-covariance, and then
    calculates the alignment. 

    Parameters
    ----------
    size : int
        Tells the low-rank SVD solver what rank to calculate. Divided by 2. 
    rank : int
        Simply holds the unmodified size of what rank to calculate. rank = size*2
    state: int
        Notes if we are using the generated (0) or reference (1) path. Usually
        modified by recursive function.
    x : tensor
        Input tensor with at least 2 dimensions. If 4-dimensional we reshape
        the paths such that the channel dimension is in the 2nd dimension and
        all other dimensions (batch x spatial) are combined into the first dimension. 
    """
    def __init__(self, size, rank):
        super().__init__()
        self.svd = SVD.apply
        self.rank = rank
        if size < rank+1:
            self.size = size//2
        else:
            self.size = rank
        #self.state=0
        self.cov = torch.zeros((rank,rank)).cuda()
        self.total = 0
        self.state=0
        self.mom =1.0

    def forward(self, x):

        if self.state == 1:
            return x
        # changing path
        x1 = x[0 : x.shape[0]//2]
        # fixed path
        x2 = x[x.shape[0]//2 : ]

        #print(x1.shape, self.rank, self.size)
        
        if x.ndim == 4:
            x1 = x1.permute(0, 2, 3, 1)
            x1 = x1.reshape((-1, x1.shape[-1]))
            
            x2 = x2.permute(0, 2, 3, 1)
            x2 = x2.reshape((-1, x2.shape[-1]))
        if self.training:
            cov = x1.T@x2
            self.total += x1.shape[0]
            #0.9
            #0.7
            self.cov = (self.mom)*cov + (1-self.mom)*self.cov.detach()
            #print("WE MOMENTUM AVG")
            temp_cov = self.cov
            temp_total = self.total
            U, S, V = self.svd(self.cov, self.size)

            V_h = V.T
            alignment = U  @ V_h
            #self.alignment = alignment
        else:
            cov = x1.T@x2
            total = x1.shape[0]
            #temp_cov = (.9)*cov + (0.1)*self.cov.detach()
            temp_cov = (self.mom)*cov + (1-self.mom)*self.cov.detach()
            temp_total = self.total
            U, S, V = self.svd(temp_cov, self.size)
            V_h = V.T
            alignment = U  @ V_h
            #self.alignment = alignment
 
        x1 =  x1@alignment

        if x.ndim == 4:
            aligned_x = x1.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)
            x_2 = x2.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)


        return torch.cat([aligned_x, x_2], dim=0)


class NewBatchNorm(nn.Module):
    def __init__(self, planes):
        super(NewBatchNorm, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.gen_dropout = 0.0
        self.ref_dropout = 0.0
        self.state = 0

    def forward(self, x):
        # changing path
        x1 = x[0 : x.shape[0]//2]
        # fixed path
        x2 = x[x.shape[0]//2 : ]

        # corresponds to always sampling generated pathway
        prob = torch.cuda.FloatTensor(1).uniform_(0, 1)
        if prob < self.gen_dropout and self.training and self.state == 0:
            x1=self.bn1(x1)
            return torch.cat([x1, x1], dim=0)
        elif prob > 1-self.ref_dropout and self.training and self.state == 0:
            x2=self.bn2(x2)
            return torch.cat([x2, x2], dim=0)
        else:
            x1=self.bn1(x1)
            x2=self.bn2(x2)
            return torch.cat([x1, x2], dim=0)

        #if prob <= 0.1 and self.training:
        #    return torch.cat([x1, x1], dim=0)
        ##elif prob >= 0.5 and self.training:
        ##    return torch.cat([x2, x2], dim=0)
        #else:
        #    return torch.cat([x1, x2], dim=0)
        #    #return torch.cat([x1, x2], dim=0)
        #x1=self.bn1(x1)
        #x2=self.bn2(x2)
        #return torch.cat([x1, x2], dim=0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.align1 = Alignment(in_planes, in_planes)
        self.bn1 = NewBatchNorm(in_planes)
        self.conv1 = NewResamplingDoubleFactConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.align2 = Alignment(planes, planes)
        self.bn2 = NewBatchNorm(planes)
        self.conv2 = NewResamplingDoubleFactConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        #self.align2 = Alignment(planes, planes)

        #self.bn3 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.shortcut = nn.Sequential()#Concat())
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                #Alignment(in_planes, in_planes),
                #self.align1,
                #self.bn1,
                NewResamplingDoubleFactConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(planes, track_running_stats=True),
            )

    def forward(self, x):
        x_align  = self.bn1(self.align1(x))
        #x_ax = self.bn1(x)
        out = F.relu(self.conv1(x_align))
        out = self.conv2(self.bn2(self.align2(out)))
        if len(self.shortcut) != 0:
            x = x_align
        out += self.shortcut(x)
        out = F.relu(out)
        #align?

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = NewResamplingDoubleFactConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = NewResamplingDoubleFactConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = NewResamplingDoubleFactConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(planes),
                NewResamplingDoubleFactConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(self.bn1(x)))
        out = F.relu(self.conv2(self.bn2(out)))
        out = self.conv3(self.bn3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = NewResamplingDoubleFactConv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.align1 = Alignment(64, 64)
        #self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.align =  Alignment(512*block.expansion, 512*block.expansion)
        #self.align =  nn.Identity()# Alignment(512*block.expansion, 512*block.expansion)
        self.align =  Alignment(512*block.expansion, 512*block.expansion)
        self.bn_final = NewBatchNorm(512*block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(self.bn_final(self.align(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class RainbowNet(nn.Module):
    def __init__(self, resnet):
        super(RainbowNet, self).__init__()
        self.resnet = resnet


    def forward(self, x):
        x = torch.cat([x, x], dim=0)
        out = self.resnet(x)
        return out



def AltAlignedResNet18():
    return RainbowNet(ResNet(BasicBlock, [2, 2, 2, 2]))




def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

#test()
