#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Nullable allows a neural network to handle "null" inputs."""

import torch
import torch.nn as nn

class Nullable(nn.Module):
    def __init__(self, module, fillmethod):
        super(Nullable, self).__init__()
        self.module = module
        self.fillmethod = fillmethod

    def forward(self, *args):
        inp, = args
        indicator = inp["indicator"]
        indices = indicator.flatten().nonzero(as_tuple=True)
        data = inp["data"]
        if indices[0].shape[0] == 0:
            partial = self.module(torch.randn(data[0:1].shape).type_as(data))
        else:
            partial = self.module(data[indices])
        output = self.fillmethod(self.training, tuple(data.shape[:1] + partial.shape[1:])).type_as(data)
        if indices[0].shape[0] > 0:
            output[indices] = partial
        return output
