import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss


class CELossWithMaxEntRegularizer(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', max_ent_factor=0.1):
        super(CELossWithMaxEntRegularizer, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.max_ent_factor = max_ent_factor

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction) + \
               self.max_ent_factor * (F.softmax(input, dim=1) * F.log_softmax(input, dim=1)).sum()
        # print('CE loss {}'.format(loss))
        # print('H loss {}'.format(loss_1))
        # return loss + loss_1



