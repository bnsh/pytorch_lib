#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""SpatialCrossMapLRN is basically my attempt to recreate
https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialCrossMapLRN
                          x_f
y_f =  -------------------------------------------------
        (k+(alpha/size) * sum_{l=l1 to l2} (x_l^2))^beta

where x_f is the input at spatial locations h,w (not shown for simplicity) and
feature map f,
    l1 corresponds to max(0,f-floor(size/2)) and
    l2 to min(F, f + floor(size/2)).

Here, F is the number of feature maps. More information can be found at
https://code.google.com/p/cuda-convnet2/wiki/LayerParams#Local_response_normalization_layer_%28across_maps%29
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class SpatialCrossMapLRN(nn.Module):
    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, *rawdata):
        inp, = rawdata

        is_batch = True
        if inp.dim() == 3:
            inp = torch.unsqueeze(inp, 0)
            is_batch = False

        channels = inp.size(1)

        channel_lrns = []

        for channel in range(0, channels):
            lower_bound = max(0, channel-math.floor(self.size/2.0))
            upper_bound = min(channels-1, channel+math.floor(self.size/2.0))

            numerator = inp.select(1, channel)
            indices = Variable(torch.arange(lower_bound, upper_bound+1).long())
            if inp.is_cuda:
                indices = indices.cuda()
            denominator = torch.pow(self.k + (self.alpha/self.size) * (torch.pow(inp.index_select(1, indices), 2)).sum(1), self.beta)
            back = (numerator/denominator).unsqueeze(1)
            channel_lrns.append(back)

        output = torch.cat(channel_lrns, 1)

        if not is_batch:
            output = output[0]

        return output
