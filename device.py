#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""Push input to the specified device"""

import torch.nn as nn

class Device(nn.Module):
    def __init__(self, device):
        super(Device, self).__init__()
        self.device = device

    def forward(self, *args):
        inps, *_ = args
        return inps.to(self.device)
