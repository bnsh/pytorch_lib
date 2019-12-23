#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""This tests the Normalize method."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorchlib.normalize import Normalize
from pytorchlib.sparsify import Sparsify

def floatrange(low, high, increment):
    curr = low
    while curr < high:
        yield curr
        curr += increment

# p and lp are widely recognizable in this context.
#pylint: disable=invalid-name
def main():
    for p in floatrange(0.5, 10, 0.5):
        lpnorm = Normalize(p=p)
        data = Variable(torch.randn(10, 10))
        lp = lpnorm(data)
        lpverify = torch.pow(lp, p).sum(1)
        assert 0.9999 <= lpverify <= 1.0001

    raw = Variable(torch.rand(10, 10))
    combined = nn.Sequential( \
        Sparsify(k=4, replacement=0), \
        Normalize(p=1) \
    )
    probs = combined(raw)
    print(probs)

if __name__ == "__main__":
    main()
