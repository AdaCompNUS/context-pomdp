"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import math
import numpy as np
import sys
from PyTorch_to_C import global_params
global_config = global_params.config


class MDN(torch.jit.ScriptModule):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    __constants__ = ['in_features', 'out_features', 'num_gaussians',
        'sigma_smoothing']
    # 
    def __init__(self, in_features = 1024, out_features = 1, num_gaussians = 5, 
        sigma_smoothing = global_config.sigma_smoothing):
        super(MDN, self).__init__()
        self.sigma_smoothing = sigma_smoothing
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        # TODO
        # self.softmax = torch.jit.trace(nn.Softmax(dim = 1), torch.randn(1, self.num_gaussians))
        # self.linear = torch.jit.trace(nn.Linear(self.in_features, self.num_gaussians), 
        #     torch.randn(1, self.in_features))
        # self.pi = torch.jit.trace(nn.Sequential(
        #     self.linear,
        #     self.softmax),
        #     torch.randn(1, self.in_features))
        self.pi = torch.jit.trace(nn.Sequential(
            nn.Linear(self.in_features, self.num_gaussians),
            nn.Softmax(dim = 1)),
            torch.randn(1, self.in_features))
        self.sigma = torch.jit.trace(nn.Linear(self.in_features, 
            self.out_features*self.num_gaussians),
            torch.randn(1, self.in_features))
        self.mu = torch.jit.trace(nn.Linear(self.in_features, 
            self.out_features*self.num_gaussians),
            torch.randn(1, self.in_features))

    @torch.jit.script_method
    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = self.sigma(minibatch)
        mu = self.mu(minibatch)
       
        sigma = torch.exp(sigma)
        sigma = torch.add(sigma, self.sigma_smoothing)

        sigma = sigma.view(-1, self.num_gaussians, self.out_features)

        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

MyModule = MDN()
MyModule.save('MDN.pt')
