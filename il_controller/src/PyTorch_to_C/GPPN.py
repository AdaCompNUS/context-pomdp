import torch
import torch.nn as nn

# import sys
#
# sys.path.append('../')
#
# print (sys.path)

from Data_processing import global_params

config = global_params.config

imsize_eg_input = 32 # example input 

# Gated Path Planning Network module
class GPPN(torch.jit.ScriptModule):
    """
    Implementation of the Gated Path Planning Network.
    """
    __constants__ = ['maze_size', 'l_i', 'l_h', 'k', 'f']
    # 5, 100, 10, 7
    def __init__(self, l_i = 5, l_h = 100, k = 10, f = 7):   
        super(GPPN, self).__init__()
        
        if config.vanilla_resnet: # In this case VIN will be disabled.
            self.output_channels = l_i 
        else:    
            self.output_channels = config.gppn_out_channels # 14
            # print(config.gppn_out_channels)
        
        self.maze_size = imsize_eg_input

        self.l_i = l_i
        self.l_h = l_h
        self.k = k       
        self.f = f

        self.hid = torch.jit.trace(nn.Conv2d(
            in_channels=self.l_i,  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True),
            torch.randn(1, self.l_i, imsize_eg_input, imsize_eg_input)
            )

        self.h0 = torch.jit.trace(nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True),
            torch.randn(1, self.l_h, imsize_eg_input, imsize_eg_input)
            )

        self.c0 = torch.jit.trace(nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True),
            torch.randn(1, self.l_h, imsize_eg_input, imsize_eg_input)
            )

        self.conv = torch.jit.trace(nn.Conv2d(
            in_channels=self.l_h,
            out_channels=1,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            bias=True),
            torch.randn(1, self.l_h, imsize_eg_input, imsize_eg_input)
            )

        self.lstm = torch.jit.trace(nn.LSTMCell(1, self.l_h), 
            (torch.randn(1 * imsize_eg_input * imsize_eg_input, 1), 
            (torch.randn(imsize_eg_input * imsize_eg_input, self.l_h), 
            torch.randn(imsize_eg_input * imsize_eg_input, self.l_h)))
            )

        self.policy = torch.jit.trace(nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.output_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False),
            torch.randn(1, self.l_h, imsize_eg_input, imsize_eg_input)
            )

    @torch.jit.script_method        
    def forward(self, X):

        # maze_size = X.size()[2] # image size

        maze_size = self.maze_size
        hid = self.hid(X)
        h0 = self.h0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)
        c0 = self.c0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)

        last_h, last_c = h0, c0
        for loop in range(self.k - 1):
            h_map = last_h.view(-1, maze_size, maze_size, self.l_h)
            h_map = h_map.transpose(3, 1)
            inp = self.conv(h_map).transpose(1, 3).contiguous().view(-1, 1)

            last_h, last_c = self.lstm(inp, (last_h, last_c))

        hk = last_h.view(-1, maze_size, maze_size, self.l_h).transpose(3, 1)
        logits = self.policy(hk)
        return logits



MyModule = GPPN()
MyModule.save('GPPN.pt')
