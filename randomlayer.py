#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""Random will simply always return random values.
WHY?! Well, it's useful to null out particular entries,
without altering overall network architecture."""

import torch
import torch.nn as nn
from torch.autograd import Variable

class Random(nn.Module):
    def __init__(self, size, mean, standard_deviation):
        super(Random, self).__init__()
        if isinstance(size, int):
            size = [size]
        self.size = size
        self.mean = mean
        self.standard_deviation = standard_deviation

    def forward(self, *rawdata):
        data, = rawdata
        data_size = list(data.size())
        output_size = [data_size[0]] + self.size
        return Variable(torch.ones(*output_size).mul_(self.standard_deviation).add_(self.mean))
