import torch
import torch.nn as nn
import math
from Data_processing import global_params
import pdb
import numpy as np
import collections

config = global_params.config

imsize_eg_input_32 = 32 # example input 
imsize_eg_input_16 = 16 # example input 

disable_bn_in_resnet = config.disable_bn_in_resnet
num_layers = config.Num_resnet_layers # 3
resnet_width = config.resnet_width # 64

expansion = 1 # expansion defined in class Block

class BasicBlock(torch.jit.ScriptModule):

    def __init__(self, in_planes = 32, planes = 32, stride=1):
        super(BasicBlock, self).__init__()
         
        self.conv1 = torch.jit.trace(nn.Conv2d(in_planes, planes, kernel_size = 3, 
            stride = stride, padding = 1, bias = False),
            torch.randn(1, in_planes, imsize_eg_input_16, imsize_eg_input_16))

        self.bn1 = torch.jit.trace(nn.BatchNorm2d(planes, track_running_stats=config.track_running_stats), 
            torch.randn(1, planes, imsize_eg_input_16, imsize_eg_input_16))
        self.relu = torch.jit.trace(nn.ReLU(inplace=True), torch.randn(2))
        
        self.conv2 = torch.jit.trace(nn.Conv2d(planes, planes, kernel_size = 3, 
            stride = 1, padding = 1, bias = False),
            torch.randn(1, planes, imsize_eg_input_16, imsize_eg_input_16))

        self.bn2 = torch.jit.trace(nn.BatchNorm2d(planes, track_running_stats=config.track_running_stats),
            torch.randn(1, planes, imsize_eg_input_16, imsize_eg_input_16))
   

    @torch.jit.script_method
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # if self.disable_bn_in_resnet == 0:
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # if self.disable_bn_in_resnet == 0:
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


class ResNetModified(torch.jit.ScriptModule):

    __constants__ = ['in_planes']
    # def __init__(self, block, layers, ip_channels=3):
    def __init__(self, layers = [2, 2, 2, 2], ip_channels = 3, resnet_width_local = 64):
        # ap_kernelsize changed from 7 to 3
        
        super(ResNetModified, self).__init__()
        self.in_planes = resnet_width_local
    
        # changing no of input channels (originally 3 in resnet)
        self.conv1 = torch.jit.trace(nn.Conv2d(ip_channels, self.in_planes, kernel_size = 7, stride = 2, padding = 3, bias = False), 
            torch.randn(1, ip_channels, imsize_eg_input_32, imsize_eg_input_32)
            )

        self.bn1 = torch.jit.trace(nn.BatchNorm2d(self.in_planes, track_running_stats = config.track_running_stats),
            torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16)
            )

        self.relu = torch.jit.trace(nn.ReLU(inplace=True), torch.randn(2))

        layer_ordererdict = collections.OrderedDict()
        layer_ordererdict['0'] = BasicBlock(self.in_planes, self.in_planes, 
            stride = 1)
        for i in range(layers[0]):
            if i == 0:
                continue
            else:
                layer_ordererdict[str(i)] = BasicBlock(self.in_planes, 
                    self.in_planes, stride = 1)
        self.layer1 = torch.jit.trace(nn.Sequential(layer_ordererdict),
            torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16))

        layer_ordererdict = collections.OrderedDict()
        layer_ordererdict['0'] = BasicBlock(self.in_planes, self.in_planes, stride = 1)
        for i in range(layers[1]):
            if i == 0:
                continue
            else:
                layer_ordererdict[str(i)] = BasicBlock(self.in_planes, self.in_planes, stride = 1)
        self.layer2 = torch.jit.trace(nn.Sequential(layer_ordererdict),
            torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16))        
       
        
        layer_ordererdict = collections.OrderedDict()
        layer_ordererdict['0'] = BasicBlock(self.in_planes, self.in_planes, stride = 1)
        for i in range(layers[2]):
            if i == 0:
                continue
            else:
                layer_ordererdict[str(i)] = BasicBlock(self.in_planes, self.in_planes, stride = 1)
        self.layer3 = torch.jit.trace(nn.Sequential(layer_ordererdict),
            torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16))
       
        
        # layer_ordererdict = collections.OrderedDict()
        # layer_ordererdict['0'] = BasicBlock(self.in_planes, self.in_planes, stride = 1)
        # for i in range(layers[3]):
        #     if i == 0:
        #         continue
        #     else:
        #         layer_ordererdict[str(i)] = BasicBlock(self.in_planes, self.in_planes, stride = 1)
        # self.layer4 = torch.jit.trace(nn.Sequential(layer_ordererdict),
        #     torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16))
        
    @torch.jit.script_method
    def forward(self, x):
        x = self.conv1(x)
        # if self.disable_bn_in_resnet == 0:
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #    x = self.layer4(x)
        return x


class ResNet_2L(torch.jit.ScriptModule):
    __constants__ = ['in_planes']

    # def __init__(self, block, layers, ip_channels=3):
    def __init__(self, layers=[2, 2, 2, 2], ip_channels=3, resnet_width_local=64):
        # ap_kernelsize changed from 7 to 3

        super(ResNet_2L, self).__init__()
        self.in_planes = resnet_width_local

        # changing no of input channels (originally 3 in resnet)
        self.conv1 = torch.jit.trace(
            nn.Conv2d(ip_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
            torch.randn(1, ip_channels, imsize_eg_input_32, imsize_eg_input_32)
            )

        self.bn1 = torch.jit.trace(nn.BatchNorm2d(self.in_planes, track_running_stats=config.track_running_stats),
                                   torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16)
                                   )

        self.relu = torch.jit.trace(nn.ReLU(inplace=True), torch.randn(2))

        layer_ordererdict = collections.OrderedDict()
        layer_ordererdict['0'] = BasicBlock(self.in_planes, self.in_planes,
                                            stride=1)
        for i in range(layers[0]):
            if i == 0:
                continue
            else:
                layer_ordererdict[str(i)] = BasicBlock(self.in_planes,
                                                       self.in_planes, stride=1)
        self.layer1 = torch.jit.trace(nn.Sequential(layer_ordererdict),
                                      torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16))

        layer_ordererdict = collections.OrderedDict()
        layer_ordererdict['0'] = BasicBlock(self.in_planes, self.in_planes, stride=1)
        for i in range(layers[1]):
            if i == 0:
                continue
            else:
                layer_ordererdict[str(i)] = BasicBlock(self.in_planes, self.in_planes, stride=1)
        self.layer2 = torch.jit.trace(nn.Sequential(layer_ordererdict),
                                      torch.randn(1, self.in_planes, imsize_eg_input_16, imsize_eg_input_16))


    @torch.jit.script_method
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

MyModule = ResNetModified()
MyModule.save('ResNetModified.pt')

