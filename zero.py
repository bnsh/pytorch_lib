#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Zero will simply always return zeros.
WHY?! Well, it's useful to null out particular entries,
without altering overall network architecture."""

import torch
import torch.nn as nn
from torch.autograd import Variable

class Zero(nn.Module):
    def __init__(self, size):
        super(Zero, self).__init__()
        if isinstance(size, int):
            size = [size]
        self.size = size

    def forward(self, *rawdata):
        data, = rawdata
        data_size = list(data.size())
        output_size = [data_size[0]] + self.size
        return Variable(torch.zeros(*output_size))
