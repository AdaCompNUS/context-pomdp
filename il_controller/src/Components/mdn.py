"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
from Components.nan_police import *
import numpy as np
import sys
from Data_processing import global_params
from Data_processing.global_params import error_handler
global_config = global_params.config

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)


def validate_sigma(sigma):
    if False:
        if np.any(sigma < 0.000001):
            sigma_entry = sigma[np.where(sigma < 0.000001)][0]
            print(sigma_entry)
            import pdb
            pdb.set_trace()
        else:
            print(sigma)


def validate_parameters(pi, sigma):
    if False:
        if np.any(sigma > 20):
            sigma_entry = sigma[np.where(sigma > 20)][0]
            print(sigma_entry)
        if np.any(sigma < -20):
            sigma_entry = sigma[np.where(sigma < -20)][0]
            print(sigma_entry)
        validate_pi(pi)


def validate_pi(pi):
    if False:
        for i in range(pi.shape[0]):
            entry = pi[i]
            if np.all(entry < 0.000001):
                pi_entry = entry[np.where(entry < 0.000001)][0]
                print(entry, pi_entry)
            if torch.sum(entry, dim=0) > 1.01:
                print(entry)
            if torch.sum(entry, dim=0) < 0.99:
                print(entry)


def validate_prob(prob, pi, mu, sigma):
    return
    # if hasinf(prob) or hasnan(prob):
    #     error_handler("inf or nan in prob")
    # if hasinf(pi) or hasnan(pi):
    #     error_handler("inf or nan in pi")
    # if hasinf(mu) or hasnan(mu):
    #     error_handler("inf or nan in mu")
    # if hasinf(sigma) or hasnan(sigma):
    #     error_handler("inf or nan in sigma")


def validate_prob1(prob, data, mu, sigma):
    return
    # validate_prob(prob, data, mu, sigma)


class MDN(nn.Module):
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
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        # self.sigma = nn.Linear(in_features, out_features*num_gaussians)

        self.sigma = nn.Sequential(nn.Linear(in_features, out_features*num_gaussians),
                                   nn.ELU(True))
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, minibatch):

        pi = self.pi(minibatch)
        sigma = self.sigma(minibatch)
        mu = self.mu(minibatch)

        # validate_parameters(pi, sigma)

        # sigma = self.safe_sigma(sigma)
        # scalar = sigma.new_empty(sigma.shape).fill_(global_config.sigma_smoothing)

        # scalar = sigma.new_empty(sigma.shape).fill_(1.0 + global_config.sigma_smoothing)

        # sigma = sigma + scalar
        sigma = torch.add(sigma, global_config.sigma_smoothing + 1.0)

        # if hasnan(sigma) or hasinf(sigma):
        #     error_handler("isnan in mdn forward")

        sigma = sigma.view(-1, self.num_gaussians, self.out_features)

        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

    def safe_sigma(self, sigma):
        if False:
            if np.any(sigma > 20):
                scalar = sigma.new_empty(sigma.shape).fill_(20)
                sigma = torch.min(sigma, scalar.expand_as(sigma))
            if np.any(sigma < -20):
                scalar = sigma.new_empty(sigma.shape).fill_(20)
                sigma = torch.max(sigma, scalar.expand_as(sigma))
        return sigma


# class MDN(nn.Module):
#     """A mixture density network layer
#     The input maps to the parameters of a MoG probability distribution, where
#     each Gaussian has O dimensions and diagonal covariance.
#     Arguments:
#         in_features (int): the number of dimensions in the input
#         out_features (int): the number of dimensions in the output
#         num_gaussians (int): the number of Gaussians per output dimensions
#     Input:
#         minibatch (BxD): B is the batch size and D is the number of input
#             dimensions.
#     Output:
#         (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
#             number of Gaussians, and O is the number of dimensions for each
#             Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
#             is the standard deviation of each Gaussian. Mu is the mean of each
#             Gaussian.
#     """
#     def __init__(self, in_features, out_features, num_gaussians):
#         super(MDN, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_gaussians = num_gaussians
#         self.pi = nn.Sequential(
#             nn.Linear(in_features, num_gaussians),
#             nn.Softmax(dim=1)
#         )
#         self.sigma = nn.Linear(in_features, out_features*num_gaussians)
#         self.mu = nn.Linear(in_features, out_features*num_gaussians)
#
#     def forward(self, minibatch):
#         pi = self.pi(minibatch)
#         sigma = self.sigma(minibatch)
#         mu = self.mu(minibatch)
#
#         validate_parameters(pi, sigma)
#
#         sigma = self.safe_sigma(sigma)
#
#         scalar = sigma.new_empty(sigma.shape).fill_(global_config.sigma_smoothing)
#
#         sigma = torch.exp(sigma) + scalar
#
#         validate_sigma(sigma)
#
#         sigma = sigma.view(-1, self.num_gaussians, self.out_features)
#
#         mu = mu.view(-1, self.num_gaussians, self.out_features)
#         return pi, sigma, mu
#
#     def safe_sigma(self, sigma):
#         if False:
#             if np.any(sigma > 20):
#                 scalar = sigma.new_empty(sigma.shape).fill_(20)
#                 sigma = torch.min(sigma, scalar.expand_as(sigma))
#             if np.any(sigma < -20):
#                 scalar = sigma.new_empty(sigma.shape).fill_(20)
#                 sigma = torch.max(sigma, scalar.expand_as(sigma))
#         return sigma


def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """

    # print("data input: ", data.shape)

    data = data.double()
    sigma = sigma.double()
    mu = mu.double()

    validate_sigma(sigma)

    data = data.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma) ** 2) / sigma
    # if hasnan(ret):
    #     print("Investigate nan: sigma = 0 count ", torch.sum(sigma == 0))
    #     print("Investigate nan: ret = nan count", torch.sum(sigma == float('nan')))

    probs = torch.prod(ret, 2)
    validate_prob1(probs, data, mu, sigma)

    return probs


def gaussian_probability_np(sigma, mu, data):

    validate_sigma(sigma)

    ret = ONEOVERSQRT2PI * np.exp(-0.5 * ((data - mu) / sigma) ** 2) / sigma

    probs = ret * 2

    return probs
#
#
# def isnan(x):
#     return x != x


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    validate_pi(pi)
    validate_sigma(sigma)

    prob = pi * gaussian_probability(sigma, mu, target).float()

    validate_prob(prob, pi, mu, sigma)

    safe_sum = torch.sum(prob, dim=1)

    safe_sum = safe_prob(safe_sum)

    # if hasinf(safe_sum) or hasnan(safe_sum):
    #     error_handler("inf or nan in safe_sum")

    nll = -torch.log(safe_sum)
    # print(" mdn loss nll: ", nll)
    return torch.mean(nll)


def safe_prob(safe_sum):

    scalar = safe_sum.new_empty(safe_sum.shape).fill_(0.000001)
    safe_sum = safe_sum + scalar

    # if sys.version_info[0] < 3:
    #     poses = safe_sum < 0.000001
    #     if np.any(poses.cpu().numpy()):
    #         scalar = safe_sum.new_empty(safe_sum.shape).fill_(0.000001)
    #         safe_sum = torch.max(safe_sum, scalar.expand_as(safe_sum))
    # else:
    #     poses = safe_sum < 0.000001
    #     if np.any(poses.cpu().numpy()):
    #         scalar = safe_sum.new_empty(safe_sum.shape).fill_(0.000001)
    #         safe_sum = torch.max(safe_sum, scalar.expand_as(safe_sum))
    # # print(" mdn loss safe_sum: ", safe_sum)
    return safe_sum


def mdn_accuracy(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The accuracy is the likelihood of the data given the MoG
    parameters.
    """
    validate_pi(pi)
    validate_sigma(sigma)

    prob = pi * gaussian_probability(sigma, mu, target).float()

    validate_prob(prob, pi, mu, sigma)

    safe_sum = torch.sum(prob, dim=1)

    safe_sum = safe_prob(safe_sum)

    accuracy = torch.mean(safe_sum)
    return accuracy


def sample_mdn(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    try:
        pi = pi.cpu()
        sigma = sigma.cpu()
        mu = mu.cpu()

        categorical = Categorical(pi)
        pis = list(categorical.sample().data)
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
        for i, idx in enumerate(pis):
            sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])

    except Exception as e:
        print("Exception when publishing commands: %s", e)
        error_handler(e)
    return sample


def sample_mdn_ml(pi, sigma, mu, component="acc"):
    """Draw samples from a MoG.
    """
    try:
        pi = pi.cpu()
        sigma = sigma.cpu()
        mu = mu.cpu()

        values, pis = pi.max(1)
        # print('indices: ', indices)

        # sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
        # for i, idx in enumerate(pis):
        #     sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])

        sample = 0.0
        if component is "acc":
            for i, idx in enumerate(pis):
                if abs(float(mu[i, idx]) - 0) < 0.1:
                    sample = 0.0
                if abs(float(mu[i, idx]) - 1) < 0.1:
                    sample = 1.0
                if abs(float(mu[i, idx]) - (-1)) < 0.1:
                    sample = -1.0
        elif component is "steer":
            for i, idx in enumerate(pis):
                sample = float(mu[i, idx])

        print('mdn_ml sample:', sample)

    except Exception as e:
        print("Exception when publishing commands: %s", e)
        error_handler(e)

    return sample