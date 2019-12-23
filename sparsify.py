#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""Sparsify will take a vector and zero out all but the top k values in the input."""

import torch
import torch.nn as nn

class Sparsify(nn.Module):
    def __init__(self, k, replacement, dim=1, descending=True):
        super(Sparsify, self).__init__()
        self.k = k
        self.replacement = replacement
        self.dim = dim
        self.descending = descending

    def forward(self, *rawdata):
        inp, = rawdata

        _, indices = torch.sort(inp, self.dim, self.descending)
        _, rindices = torch.sort(indices, self.dim, False)
        returnvalue = inp.clone()
        returnvalue[rindices >= self.k] = self.replacement

        return returnvalue
