import torch
import torch.nn as nn
from Data_processing import global_params
import pdb

config = global_params.config
from Components.nan_police import *


# Gated Path Planning Network module
class GPPN(nn.Module):
    """
    Implementation of the Gated Path Planning Network.
    """
    def __init__(self, args, l_i=None, out_planes=config.gppn_out_channels):
        super(GPPN, self).__init__()

        self.output_channels = out_planes  # 1

        if l_i is not None:
            self.l_i = l_i
        else:
            self.l_i = args.l_i

        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f

        self.hid = nn.Conv2d(
            in_channels=self.l_i,  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)

        self.h0 = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)

        self.c0 = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)

        self.conv = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=1,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            bias=True)

        self.lstm = nn.LSTMCell(1, self.l_h)

        self.policy = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.output_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        param.data.zero_()
                    elif 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, X):
        maze_size = X.size()[2] # image size

        hid = self.hid(X)
        h0 = self.h0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)
        c0 = self.c0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)

        last_h, last_c = h0, c0
        for loop in range(0, self.k - 1):
            h_map = last_h.view(-1, maze_size, maze_size, self.l_h)
            h_map = h_map.transpose(3, 1)
            inp = self.conv(h_map).transpose(1, 3).contiguous().view(-1, 1)

            last_h, last_c = self.lstm(inp, (last_h, last_c))

        hk = last_h.view(-1, maze_size, maze_size, self.l_h).transpose(3, 1)
        logits = self.policy(hk)
        return logits

