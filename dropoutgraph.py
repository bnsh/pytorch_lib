#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This should be an graph dropout."""

import torch

#pylint: disable=invalid-name
def dropoutgraph(inp, p=0.5, channels=-1, training=False, inplace=False):
    if not isinstance(channels, list):
        channels = [channels]
    if training:
        mask_shape = list(inp.shape)

        for channel in channels:
            if channel == -1:
                channel = len(mask_shape)-1
            mask_shape[channel] = 1

        mask = torch.zeros(mask_shape).bernoulli(1-p).type_as(inp)
        if inplace:
            inp *= mask
        else:
            inp = inp * mask

    return inp
#pylint: enable=invalid-name
