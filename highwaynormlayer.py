#! /usr/bin/python3
# vim: expandtab shiftwidth=4 tabstop=4

"""HighwayNormLayer implements the highway layer as specified in https://arxiv.org/pdf/1505.00387.pdf, it just adds a LayerNorm to the end of it as well."""

import torch
import torch.nn as nn

# pylint: disable=too-many-instance-attributes
class HighwayNormLayer(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(self, num_layers, width, transfer_class, bias, dropout):
        super(HighwayNormLayer, self).__init__()
        tfeps = 1e-12 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315 Just for TF compatibility
        self.num_layers = num_layers
        self.width = width
        self.transfer_class = transfer_class
        self.bias = bias
        self.dropout = dropout

        self.dropouts = [nn.Dropout(dropout) for i in range(0, num_layers)]
        self.gate_linears = [nn.Linear(width, width) for i in range(0, num_layers)]
        self.transfer_linears = [nn.Linear(width, width) for i in range(0, num_layers)]
        self.transfer_functions = [transfer_class() for i in range(0, num_layers)]
        self.layer_norms = [nn.LayerNorm(width, eps=tfeps) for i in range(0, num_layers)]

        for i in range(0, num_layers):
            setattr(self, "dropouts[%d]" % (i), self.dropouts[i])
            setattr(self, "gate_linears[%d]" % (i), self.gate_linears[i])
            setattr(self, "transfer_linears[%d]" % (i), self.transfer_linears[i])
            setattr(self, "transfer_functions[%d]" % (i), self.transfer_functions[i])
            setattr(self, "layer_norms[%d]" % (i), self.layer_norms[i])
    # pylint: enable=too-many-arguments

    def forward(self, *data):
        return_value, = data

        for i in range(0, self.num_layers):
            droppedout = self.dropouts[i](return_value)
            gate_inputs = self.gate_linears[i](droppedout)
            transfer_inputs = self.transfer_linears[i](droppedout)
            sigmoid_values = torch.sigmoid(gate_inputs + self.bias)
            raw_value = (sigmoid_values * self.transfer_functions[i](transfer_inputs)) + ((1-sigmoid_values) * return_value)
            return_value = self.layer_norms[i](raw_value)

        return 1 * return_value
# pylint: enable=too-many-instance-attributes
