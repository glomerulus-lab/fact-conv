'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_modules import ResamplingDoubleFactConv2d
from align import Alignment


class Concat(nn.Module):
    def forward(self, x):
        return torch.cat([x, x], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.align1 = Alignment(in_planes, in_planes)
        # rainbow networks double the batchnorm channel size
        self.bn1 = nn.BatchNorm2d(in_planes*2, track_running_stats=True)
        self.conv1 = ResamplingDoubleFactConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.align2 = Alignment(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes*2, track_running_stats=True)
        self.conv2 = ResamplingDoubleFactConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        #self.align2 = Alignment(planes, planes)

        #self.bn3 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.shortcut = nn.Sequential()#Concat())
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                #Alignment(in_planes, in_planes),
                #self.align1,
                #self.bn1,
                ResamplingDoubleFactConv2d(in_planes, self.expansion*planes,
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
        self.conv1 = ResamplingDoubleFactConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ResamplingDoubleFactConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = ResamplingDoubleFactConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(planes),
                ResamplingDoubleFactConv2d(in_planes, self.expansion*planes,
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

        self.conv1 = ResamplingDoubleFactConv2d(3, 64, kernel_size=3,
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
        self.bn_final = nn.BatchNorm2d(512*block.expansion*2, track_running_stats=True)
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

import copy
class RainbowNet(nn.Module):
    def __init__(self, resnet):
        super(RainbowNet, self).__init__()
        # we don't want to mix outputs, so we move the linear layer away from
        # the network in order to grab the features, split channel in half, 
        # then feed into linear layer
        self.linear = copy.deepcopy(resnet.linear)
        resnet.linear = nn.Identity()
        self.resnet = resnet


    def forward(self, x):
        # concat batch along channel dim
        x = torch.cat([x, x], dim=1)
        # feed into generated and reference networks
        out = self.resnet(x)
        # get generated path features
        y_1 = out[:, 0:out.shape[1]//2]
        # get reference path features
        y_2 = out[:, out.shape[1]//2:]
        # concatenate along batch dim
        out = torch.cat([y_1, y_2],dim=0)
        # get output logits. first half is generated network, 2nd half is
        # reference network
        out = self.linear(out)
        return out




def AlignedResNet18():
    return RainbowNet(ResNet(BasicBlock, [2, 2, 2, 2]))

def AlignedResNet9():
    return ResNet(BasicBlock, [1, 1, 1, 1])



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
