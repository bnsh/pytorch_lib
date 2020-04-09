#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This module will implement a variational autoencoder _layer_"""

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_features, out_features):
        super(VAE, self).__init__()
        self.fc_mean = nn.Linear(in_features, out_features)
        self.fc_logvariance = nn.Linear(in_features, out_features)
        self.mean = torch.zeros(1, 1)
        self.logvariance = torch.zeros(1, 1)

    #pylint: disable=arguments-differ
    def forward(self, inps):
        self.mean = self.fc_mean(inps)
        self.logvariance = self.fc_logvariance(inps)
        std = torch.exp(self.logvariance/2.0)

        if self.training:
            random = torch.randn(self.mean.shape).type_as(inps)
        else:
            random = torch.zeros(self.mean.shape).type_as(inps)

        retval = self.mean + random * std
        return retval
    #pylint: enable=arguments-differ

    def loss(self):
        kl_loss = -0.5 * torch.sum(1.0 + self.logvariance - (self.mean * self.mean) - torch.exp(self.logvariance), dim=1)
        return kl_loss

def main():
    pass

if __name__ == "__main__":
    main()
