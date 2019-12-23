#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""This module will return a RReLU that has a zero mean.
   (as opposed to the usual RReLU that has a nonzero mean.)
"""

import math
import torch.nn as nn

class ZeroMeanRReLU(nn.Module):
    def __init__(self, loval=1.0/8.0, hival=1.0/3.0):
        super(ZeroMeanRReLU, self).__init__()
        self.loval = loval
        self.hival = hival
        self.rrelu = nn.RReLU(self.loval, self.hival)
        #FullSimplify[
        #	Integrate[
        #        PDF[NormalDistribution[0, 1], x] *
        #        PDF[UniformDistribution[{a, b}], r] x r,
        #        {r, a, b},
        #        {x, -Infinity, 0},
        #        Assumptions -> {b > a, Im[b] == 0, Im[a] == 0}] +
        #	Integrate[PDF[NormalDistribution[0, 1], x] x, {x, 0, Infinity}]
        #]
        self.adjust = (2.0 - self.loval - self.hival)/(4 * math.sqrt(math.pi * 2.0))

    def forward(self, *inp):
        retval = self.rrelu(*inp) - self.adjust
        return retval
