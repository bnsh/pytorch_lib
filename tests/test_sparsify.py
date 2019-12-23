#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This tests the Sparsify module"""

import torch
from torch.autograd import Variable
from pytorchlib.sparsify import Sparsify

def main():
    sparsify = Sparsify(k=2, replacement=-1)
    data = Variable(torch.arange(0, 10).resize_(1, 10))
    print(data)
    print(sparsify(data))

if __name__ == "__main__":
    main()
