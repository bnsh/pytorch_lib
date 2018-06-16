#! /usr/bin/python

"""HighwayLSTMLayer implements basically HighwayLayers except with the
   tied to an LSTMCell. There _is_ a paper on this:
	https://arxiv.org/pdf/1510.08983.pdf

   But, this is _NOT_ that! It's my own concoction! I've really just literally replaced
   the HighwayLayer's transfer function with an LSTMCell, and am just _experimenting_!
   Caveat Emptor and all that!!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
		# XTODO: THIS IS ALL A MESS! A LSTM CELL IS ONLY DOING ONE TIME STEP,
		# AS OPPOSED TO A LSTM, so for the HighwayLSTMLayer, we'll have to do all that
		# ourselves.
		return_value, = data
		seqsz = return_value.size(0)
		batchsz = return_value.size(1)
		datasz = return_value.size(2)
		h_0 = Variable(torch.zeros(batchsz, self.width), requires_grad=False).type_as(return_value)
		c_0 = Variable(torch.zeros(batchsz, self.width), requires_grad=False).type_as(return_value)

		# So, first resize to be linear. Is this valid?
		resized = return_value.reshape(seqsz * batchsz, datasz)

		for i in xrange(0, self.num_layers):
			droppedout = self.dropouts[i](resized)
			gate_inputs = self.gate_linears[i](droppedout)
			rnn_inputs = droppedout.reshape(seqsz, batchsz, self.width)
			print rnn_inputs.size(), h_0.size(), c_0.size()
			transfer_inputs = self.lstms[i](rnn_inputs, (h_0, c_0))[0] # are the hidden states
			transfer_inputs = transfer_inputs.reshape(seqsz * batchsz, self.width)
			sigmoid_values = F.sigmoid(gate_inputs + self.bias)
			resized = (sigmoid_values * transfer_inputs) + ((1-sigmoid_values) * resized)

		return_value = resized.reshape(seqsz, batchsz, self.width)

		return return_value
