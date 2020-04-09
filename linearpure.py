#! /usr/bin/env python3

"""
    This file simply exists because I had LinearNorm everywhere, and wanted to remove layernormalization.
    I could just change all the code to nn.Linear, but I want to be able to revert back to LayerNorm if
    need be, so I'm just going to change all the LinearNorms to LinearPure's and see what happens.
"""

from collections import OrderedDict
import torch
import torch.nn as nn

class LinearPure(nn.Sequential):
    #pylint: disable=too-many-arguments,unused-argument
    def __init__(self, in_features, out_features, bias=True, eps=1e-05, elementwise_affine=True):
        super(LinearPure, self).__init__(
         OrderedDict([
          ("linear", nn.Linear(in_features, out_features, bias)),
         ])
        )
    #pylint: enable=too-many-arguments,unused-argument

def main():
    batchsz = 256
    in_features = 2
    out_features = 8

    data = torch.randn(batchsz, in_features)
    linearnorm = LinearPure(in_features, out_features)
    out = linearnorm(data)
    print(out)

if __name__ == "__main__":
    main()
