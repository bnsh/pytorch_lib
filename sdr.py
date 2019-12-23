#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This class _should_ take a model, and return a container model that will do
   Stochastic Delta Rule on the model transparently. This is from:
   https://arxiv.org/abs/1808.03578
"""

import torch
import torch.nn as nn
import numpy as np

#pylint: disable=too-many-instance-attributes
class SDR(nn.Module):
    #pylint: disable=too-many-arguments
    def __init__(self, model, alpha=0.25, beta=0.2, zeta=0.99, n_blocks=2, zeta_ratio=0.9):
        super(SDR, self).__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.zeta_orig = zeta
        self.zeta = zeta
        self.n_blocks = n_blocks
        self.zeta_ratio = zeta_ratio

        self.means = {\
            name: param.data.min() + (param.data.max() - param.data.min()) * torch.rand(param.data.shape) \
            for name, param in self.model.named_parameters() \
        }
        self.sds = {\
            name: (np.sqrt(2.0 / np.product(param.shape)) * 0.5 * \
                torch.rand(param.data.shape)).type_as(param.data) \
            for name, param in self.model.named_parameters() \
        }
    #pylint: enable=too-many-arguments

    #pylint: disable=too-many-locals
    def forward(self, *args):
        inp = args[0]
        # So, first, we need to update the standard deviations.
        if self.training:
            self._swap_sample_in()

        retval = self.model(inp)

        if self.training:
            self._swap_sample_out()

        return retval
    #pylint: enable=too-many-locals

    def _swap_sample_in(self):
        length = len(list(self.model.parameters()))
        for paramidx, (name, param) in enumerate(self.model.named_parameters()):
            # So, we need to divide the length parameters into
            # n_blocks.
            frac = int(self.n_blocks * (1-float(paramidx)/length))
            zeta_ = self.zeta * self.zeta_ratio ** frac
            if param.grad is not None:
                self.sds[name] = zeta_ * (torch.abs(self.beta * param.grad) + self.sds[paramidx])
            else:
                self.sds[name] = zeta_ * self.sds[name]

        for paramidx, (name, param) in enumerate(self.model.named_parameters()):
            self.means[name].copy_(param.data)
            param.data.copy_(torch.distributions.Normal(param.data, self.sds[name]).sample())

    def _swap_sample_out(self):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.means[name])

    def update_zeta(self, epoch):
        self.zeta = min(0.01, self.zeta_orig * np.power(np.e, -(0.1 * epoch)))
#pylint: disable=too-many-instance-attributes

def main():
    data = torch.ones((17, 5)).cpu()
    lin = nn.Linear(5, 3).cpu()
    torch.nn.init.eye_(lin.weight)
    torch.nn.init.constant_(lin.bias, 0)
    sdrlin = SDR(lin)

    print(sdrlin(data))

if __name__ == "__main__":
    main()
