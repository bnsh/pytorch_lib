#! /usr/bin/python

"""
	ResidualLayer implements the residual layer as specified in https://arxiv.org/pdf/1512.03385.pdf
"""

import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=too-many-instance-attributes
class ResidualLayer(nn.Module):
	def __init__(self, num_layers, width, transfer_class, dropout):
		super(ResidualLayer, self).__init__()
		self.num_layers = num_layers
		self.width = width
		self.transfer_class = transfer_class
		self.dropout = dropout

		self.linear_1 = [nn.Linear(width, width) for i in xrange(0, num_layers)]
		self.linear_2 = [nn.Linear(width, width) for i in xrange(0, num_layers)]
		self.transfer_functions = [transfer_class() for i in xrange(0, num_layers)]

		for i in xrange(0, num_layers):
			setattr(self, "linear_1[%d]" % (i), self.linear_1[i])
			setattr(self, "linear_2[%d]" % (i), self.linear_2[i])
			setattr(self, "transfer_functions[%d]" % (i), self.transfer_functions[i])
	def forward(self, *data):
		return_value, = data

		for i in xrange(0, self.num_layers):
			inp = return_value

			return_value = self.linear_1[i](return_value)
			return_value = self.transfer_functions[i](return_value)
			return_value = F.dropout(return_value, p=self.dropout, training=self.training)
			return_value = self.linear_2[i](return_value)

			return_value = return_value + inp

		return return_value
# pylint: enable=too-many-instance-attributes
