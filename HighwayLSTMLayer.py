#! /usr/bin/python

"""HighwayLSTMLayer implements basically HighwayLayers except with the 
   tied to an LSTMCell. There _is_ a paper on this:
	https://arxiv.org/pdf/1510.08983.pdf

   But, this is _NOT_ that! It's my own concoction! I've really just literally replaced
   the HighwayLayer's transfer function with an LSTMCell, and am just _experimenting_!
   Caveat Emptor and all that!!
"""

import torch.nn as nn
import torch.nn.functional as F

class HighwayLSTMLayer(nn.Module):
	# pylint: disable=too-many-instance-attributes
	def __init__(self, num_layers, width, bias, dropout):
		# pylint: disable=too-many-arguments
		super(HighwayLSTMLayer, self).__init__()
		self.num_layers = num_layers
		self.width = width
		self.bias = bias
		self.dropout = dropout

		self.dropouts = [nn.Dropout(dropout) for i in xrange(0, num_layers)]
		self.gate_linears = [nn.Linear(width, width) for i in xrange(0, num_layers)]
		self.lstms = [nn.LSTMCell(width, width) for i in xrange(0, num_layers)]

		for i in xrange(0, num_layers):
			setattr(self, "dropouts[%d]" % (i), self.dropouts[i])
			setattr(self, "gate_linears[%d]" % (i), self.gate_linears[i])
			setattr(self, "lstms[%d]" % (i), self.lstms[i])

	def forward(self, *data):
		return_value, = data

		for i in xrange(0, self.num_layers):
			droppedout = self.dropouts[i](return_value)
			gate_inputs = self.gate_linears[i](droppedout)
			transfer_inputs = self.lstms[i](droppedout)
			sigmoid_values = F.sigmoid(gate_inputs + self.bias)
			return_value = (sigmoid_values * transfer_inputs) + ((1-sigmoid_values) * return_value)

		return return_value
