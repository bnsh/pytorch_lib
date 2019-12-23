#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This tests the HighwayLayer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorchlib.highwaylayer import HighwayLayer

def report(module, grad_input, grad_output):
    print(module, len(grad_input), len(grad_output))
    print(grad_input)

def main():
    net = HighwayLayer(1, 2, nn.RReLU, -0.8, 0.5)
    net.register_backward_hook(report)

    data = Variable(torch.randn(4, 2))
    target = Variable(torch.zeros(4, 2))

    output = net(data)
    loss = F.mse_loss(output, target)
    loss.backward()

if __name__ == "__main__":
    main()
