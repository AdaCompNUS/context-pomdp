import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import pdb
from Components import *
#from VIN import *
#from resnet import *
from argparse import Namespace
''' Input Tensor is of the form ==> BatchSize * Ped * map/goal/hist * Width * height
'''


class drive_net(nn.Module):

    def __init__(self, config):
        super(drive_net, self).__init__()
        self.ped_gppn = VIN(config)
        self.car_gppn = VIN(config)
        #using 0 and 1 instead of global config variables
        self.no_input_resnet = 3 * (0 + 1)
        #imsize should come from globalconfig, this is coming from cmdline config
        self.resnet = ResNet(BasicBlock, [1, 1, 1, 1], fnsize=config.imsize / 32,
                             ip_channels=self.no_input_resnet)
        self.ped_resnet = BasicBlock(3, 3)
        self.car_resnet = BasicBlock(3, 3)
        self.drop_o = nn.Dropout()
        self.value_head = nn.Linear(in_features=1000,
                                    out_features=1,
                                    bias=True)
        self.acc_head = nn.Linear(in_features=1000,
                                  out_features=1,
                                  bias=True)
        #add no of bins argument in global config to replace 14
        self.ang_head = nn.Linear(in_features=1000,
                                  out_features=14,
                                  bias=True)
        # Doing Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=np.sqrt(2))

    def forward(self, X, config):
        # Input Tensor is of the form ==> BatchSize * Agent * map/goal/hist * Width * height
        # switches batch and agent dim in order to iterate over the agent dim
        reshape_X = X.permute(1, 0, 2, 3, 4)
        #
        value_data = reshape_X[:, :, 0:2, :, :]
        hist_data = reshape_X[:, :, 2:, :, :]
        # Container for the ouput value images
        #  find a way to avoid ugly indexing!
        outputs = X.new_empty(0 + 1,
                              X.shape[0], 3, 1, config.imsize, config.imsize).fill_(0)
        # 3 is the Value image and the 2 hist layers (idx 0 is value image) #adding 1 extra dimension because VIN produces single channel output
        outputs[:, :, 1:, 0, :, :] = hist_data
        for idx in range(0 + 1):
            if idx < 0:
                outputs[idx, :, 0, :, :, :] = self.ped_gppn(
                    value_data[idx], config)
                outputs[idx, :, :, 0, :, :] = self.ped_resnet(
                    outputs[idx, :, :, 0, :, :].clone())
            else:
                outputs[idx, :, 0, :, :, :] = self.car_gppn(
                    value_data[idx], config)
                outputs[idx, :, :, 0, :, :] = self.car_resnet(
                    outputs[idx, :, :, 0, :, :].clone())
        outputs = outputs.squeeze_(dim=3)
        outputs = (outputs.permute(1, 0, 2, 3, 4)).contiguous()
        outputs = outputs.view(outputs.size(
            0), outputs.size(1) * outputs.size(2), outputs.size(3), outputs.size(4))
        res = self.resnet(outputs)
        res = self.drop_o(res)
        return self.value_head(res), self.acc_head(res), self.ang_head(res)


# dummy_net = drive_net(config)
# X = Variable(torch.randn([config.batch_size, 0 +
#                           1, 4, config.imsize, config.imsize]))
# val, pol = dummy_net.forward(X, config)
# pdb.set_trace()
