#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Actually, this test suggests that beyond a single layer neural network,
   ZeroMeanRReLU is rather useless. Worse, it may even be harmful."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorchlib.zeromeanrrelu import ZeroMeanRReLU

def main():
    data = Variable(torch.randn(1024, 1024))

    module = nn.Sequential()
    for idx in range(0, 8):
        module.add_module("linear_%d" % (idx), nn.Linear(1024, 1024))
        module.add_module("transfer_%d" % (idx), ZeroMeanRReLU())

    print(torch.mean(module(data)))

if __name__ == "__main__":
    main()
