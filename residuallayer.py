#! /usr/bin/python
# vim: expandtab shiftwidth=4 tabstop=4

"""
    ResidualLayer implements the residual layer as specified in https://arxiv.org/pdf/1512.03385.pdf
"""

import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=too-many-instance-attributes
class ResidualLayer(nn.Module):
    def __init__(self, num_layers, width, transfer_class, dropout):
        super(ResidualLayer, self).__init__()
        tfeps = 1e-12 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315 Just for TF compatibility
        self.num_layers = num_layers
        self.width = width
        self.transfer_class = transfer_class
        self.dropout = dropout

        self.linear_1 = [nn.Linear(width, width) for i in range(0, num_layers)]
        self.layer_norms_1 = [nn.LayerNorm(width, eps=tfeps) for i in range(0, num_layers)]
        self.transfer_functions_1 = [transfer_class() for i in range(0, num_layers)]

        self.linear_2 = [nn.Linear(width, width) for i in range(0, num_layers)]
        self.layer_norms_2 = [nn.LayerNorm(width, eps=tfeps) for i in range(0, num_layers)]
        self.transfer_functions_2 = [transfer_class() for i in range(0, num_layers)]

        for i in range(0, num_layers):
            setattr(self, "linear_1[%d]" % (i), self.linear_1[i])
            setattr(self, "layer_norms_1[%d]" % (i), self.layer_norms_1[i])
            setattr(self, "transfer_functions_1[%d]" % (i), self.transfer_functions_1[i])

            setattr(self, "linear_2[%d]" % (i), self.linear_2[i])
            setattr(self, "layer_norms_2[%d]" % (i), self.layer_norms_2[i])
            setattr(self, "transfer_functions_2[%d]" % (i), self.transfer_functions_2[i])

    def forward(self, *data):
        return_value, = data

        for i in range(0, self.num_layers):
            inp = return_value

            return_value = self.linear_1[i](return_value)
            return_value = self.layer_norms_1[i](return_value)
            return_value = self.transfer_functions_1[i](return_value)

            return_value = self.linear_2[i](return_value)
            return_value = self.layer_norms_2[i](return_value)

            return_value = F.dropout(return_value, p=self.dropout, training=self.training)

            return_value = return_value + inp

            return_value = self.transfer_functions_2[i](return_value)


        return 1 * return_value
# pylint: enable=too-many-instance-attributes
