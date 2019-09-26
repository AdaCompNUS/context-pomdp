import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import sys

#sys.path.append("")

from Components import resnet


class VIN(nn.Module):

    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        #defualt h in VIN
        # self.h = nn.Conv2d(in_channels=config.l_i,
        #                    out_channels=config.l_h,
        #                    kernel_size=(3, 3),
        #                    stride=1, padding=1,
        #                    bias=True)
        # customized h by panpan
        self.h = nn.Sequential(
                    nn.Conv2d(in_channels=config.l_i,
                           out_channels=config.l_h,
                           kernel_size=(5, 5),
                           stride=1, padding=2,
                           bias=True),
                    resnet.BasicBlock(inplanes=config.l_h, planes=config.l_h),
                    # nn.Conv2d(in_channels=config.l_i,
                    #        out_channels=config.l_h/4,
                    #        kernel_size=(3, 3),
                    #        stride=1, padding=1,
                    #        bias=True),
                    # nn.Conv2d(in_channels=config.l_h/4,
                    #        out_channels=config.l_h,
                    #        kernel_size=(3, 3),
                    #        stride=1, padding=1,
                    #        bias=True),
                    # resnet.BasicBlock(inplanes=config.l_i, planes=config.l_i),
                    # nn.Conv2d(in_channels=config.l_i,
                    #        out_channels=config.l_h/4,
                    #        kernel_size=(5, 5),
                    #        stride=1, padding=2,
                    #        bias=True),
                    # resnet.BasicBlock(inplanes=config.l_h/4, planes=config.l_h/4),
                    # nn.Conv2d(in_channels=config.l_h/4,
                    #        out_channels=config.l_h,
                    #        kernel_size=(5, 5),
                    #        stride=1, padding=2,
                    #        bias=True),
                    # resnet.BasicBlock(inplanes=config.l_h, planes=config.l_h),
                    # resnet.BasicBlock(inplanes=config.l_i, planes=config.l_i),
                    # nn.Conv2d(in_channels=config.l_i,
                    #        out_channels=config.l_h/4,
                    #        kernel_size=(3, 3),
                    #        stride=1, padding=1,
                    #        bias=True),
                    # nn.Conv2d(in_channels=config.l_h/4,
                    #        out_channels=config.l_h,
                    #        kernel_size=(3, 3),
                    #        stride=1, padding=1,
                    #        bias=True),
                )

        self.r = nn.Conv2d(in_channels=config.l_h,
                           out_channels=1,
                           kernel_size=(1, 1),
                           stride=1, padding=0,
                           bias=False)
        self.q = nn.Conv2d(in_channels=1,
                           out_channels=config.l_q,
                           kernel_size=(3, 3), # changged from 3 to 5
                           stride=1, padding=1, # changed from 1 to 2
                           bias=False)
        self.w = Parameter(torch.zeros(
            config.l_q, 1, 3, 3), requires_grad=True) # changed from 3,3 to 5,5
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, X, config):
        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            q = F.conv2d(torch.cat([r, v], 1),
                         torch.cat([self.q.weight, self.w], 1),
                         stride=1,
                         padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = F.conv2d(torch.cat([r, v], 1),
                     torch.cat([self.q.weight, self.w], 1),
                     stride=1,
                     padding=1)
        v, _ = torch.max(q, dim=1, keepdim=True)
        return v
