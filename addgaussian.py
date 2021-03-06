#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""AddGaussian adds Gaussian noise to the input"""

import torch.nn as nn
from torch.autograd import Variable

class AddGaussian(nn.Module):
    # pylint: disable=too-few-public-methods
    def __init__(self, mean, sigma):
        super(AddGaussian, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, *rawdata):
        data, = rawdata
        if self.training:
            noise = Variable(data.detach().clone().normal_(self.mean, self.sigma), requires_grad=False)
            return data + noise
        return 1.0 * data
