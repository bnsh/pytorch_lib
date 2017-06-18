#! /usr/bin/python

"""HighwayLayer implements the highway layer as specified in https://arxiv.org/pdf/1505.00387.pdf"""

import torch.nn as nn
import torch.nn.functional as F

class HighwayLayer(nn.Module):
	# pylint: disable=too-many-instance-attributes
	def __init__(self, num_layers, width, transfer, bias, dropout):
		# pylint: disable=too-many-arguments
		super(HighwayLayer, self).__init__()
		self.num_layers = num_layers
		self.width = width
		self.transfer = transfer
		self.bias = bias
		self.dropout = dropout

		self.dropouts = [nn.Dropout(dropout) for i in xrange(0, num_layers)]
		self.gate_linears = [nn.Linear(width, width) for i in xrange(0, num_layers)]
		self.transfer_linears = [nn.Linear(width, width) for i in xrange(0, num_layers)]
		self.transfer_functions = [transfer for i in xrange(0, num_layers)]

		for i in xrange(0, num_layers):
			setattr(self, "dropouts[%d]" % (i), self.dropouts[i])
			setattr(self, "gate_linears[%d]" % (i), self.gate_linears[i])
			setattr(self, "transfer_linears[%d]" % (i), self.transfer_linears[i])
			setattr(self, "transfer_functions[%d]" % (i), self.transfer_functions[i])

	def forward(self, *data):
		return_value, = data

		for i in xrange(0, self.num_layers):
			droppedout = self.dropouts[i](return_value)
			gate_inputs = self.gate_linears[i](droppedout)
			transfer_inputs = self.transfer_linears[i](droppedout)
			sigmoid_values = F.sigmoid(gate_inputs + self.bias)
			return_value = (sigmoid_values * self.transfer_functions[i](transfer_inputs)) + ((1-sigmoid_values) * return_value)

		return return_value
