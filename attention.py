#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This class will implement attention. I intend to make it usable on it's own
   and not _require_ a RNN.

    So, the idea, is that we'll take a hidden state _and_ a _set_ of input states.
    we will concatenate the hidden state with each of the input states and get
    a softmax out of there, then we'll multiply them by the input states to get
    our output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, inpsz, hiddensz):
        super(Attention, self).__init__()
        self.inpsz = inpsz
        self.hiddensz = hiddensz
        self.attention = nn.Linear((inpsz+hiddensz), 1)
        self.focus = None

    def forward(self, *args):
        if len(args[0]) == 2:
            (inp, hidden), *_ = args
        else:
            inp, *_ = args
            hidden = torch.zeros(inp.shape[0], self.hiddensz).type_as(inp)
        inpbatchsz, inpseqsz, inpsz_ = inp.shape
        hiddenbatchsz, hiddensz_ = hidden.shape

        assert inpbatchsz == hiddenbatchsz
        assert inpsz_ == self.inpsz
        assert hiddensz_ == self.hiddensz

        # First, let's reshape the hidden to have a "seq"
        hidden = hidden.reshape(hiddenbatchsz, 1, hiddensz_).repeat(1, inpseqsz, 1)
        concat = torch.cat([inp, hidden], dim=-1).reshape((inpbatchsz*inpseqsz), (inpsz_+hiddensz_))
        self.focus = F.softmax(self.attention(concat).reshape(inpbatchsz, inpseqsz), dim=-1)
        result = (self.focus.reshape(inpbatchsz, inpseqsz, 1) * inp).sum(dim=1)
        return result
