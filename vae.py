#! /usr/bin/env python3
# vim: expandtab shiftwidth=4 tabstop=4

"""This module will implement a variational autoencoder _layer_"""

import torch
import torch.nn as nn
from .linearnorm import LinearNorm

class VAE(nn.Module):
    def __init__(self, in_features, out_features):
        super(VAE, self).__init__()
        self.fc_mean = LinearNorm(in_features, out_features)
        self.fc_logvariance = LinearNorm(in_features, out_features)
        self.mean = None
        self.logvariance = None

    def forward(self, *raw):
        inps, *_ = raw

        self.mean = self.fc_mean(inps)
        self.logvariance = self.fc_logvariance(inps)
        std = torch.exp(self.logvariance/2.0)

        random = torch.randn(self.mean.shape).type_as(inps)
        return self.mean + random * std

    def loss(self):
        kl_loss = -0.5 * torch.sum(1.0 + self.logvariance - (self.mean * self.mean) - torch.exp(self.logvariance), axis=1)
        return kl_loss

def main():
    pass

if __name__ == "__main__":
    main()
