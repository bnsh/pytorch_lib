#! /usr/bin/env python3

"""This is purely just a decorative class...
   Basically, I'm so often doing Linear followed by
   LayerNorm, so I figure I'll just put them into a
   single class like this."""

from collections import OrderedDict
import torch
import torch.nn as nn

class LinearNorm(nn.Sequential):
    #pylint: disable=too-many-arguments
    def __init__(self, in_features, out_features, bias=True, eps=1e-05, elementwise_affine=True):
        super(LinearNorm, self).__init__(
         OrderedDict([
          ("linear", nn.Linear(in_features, out_features, bias)),
          ("layer_norm", nn.LayerNorm(out_features, eps, elementwise_affine)),
         ])
        )
    #pylint: enable=too-many-arguments

def main():
    batchsz = 7
    in_features = 11
    out_features = 13

    data = torch.randn(batchsz, in_features)
    linearnorm = LinearNorm(in_features, out_features)
    out = linearnorm(data)
    print(out.shape)

if __name__ == "__main__":
    main()
