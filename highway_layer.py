#! /usr/bin/python

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from xavier import xavier

class highway_layer(nn.Module):
	def __init__(self, num_layers, sz, transfer, bias, dropout):
		super(highway_layer, self).__init__()
		self.num_layers = num_layers
		self.sz = sz
		self.transfer = transfer
		self.bias = bias
		self.dropout = dropout

		self.dropouts = [nn.Dropout(dropout) for i in xrange(0, num_layers)]
		self.gate_linears = [xavier(nn.Linear(sz, sz)) for i in xrange(0, num_layers)]
		self.transfer_linears = [xavier(nn.Linear(sz, sz)) for i in xrange(0, num_layers)]
		self.transfer_functions = [transfer for i in xrange(0, num_layers)]

		for i in xrange(0, num_layers):
			setattr(self, "dropouts[%d]" % (i), self.dropouts[i])
			setattr(self, "gate_linears[%d]" % (i), self.gate_linears[i])
			setattr(self, "transfer_linears[%d]" % (i), self.transfer_linears[i])
			setattr(self, "transfer_functions[%d]" % (i), self.transfer_functions[i])

	def forward(self, data):
		rv = data

		for i in xrange(0, self.num_layers):
			droppedout = self.dropouts[i](rv)
			gate_inputs = self.gate_linears[i](droppedout)
			transfer_inputs = self.transfer_linears[i](droppedout)
			sigmoid_values = F.sigmoid(gate_inputs + self.bias)
			rv = (sigmoid_values * self.transfer_functions[i](transfer_inputs)) + ((1-sigmoid_values) * rv)

		return rv

def main():
	hl = highway_layer(1,16,nn.RReLU(), 0, 0)
	hl.train(False)
	data = Variable(torch.Tensor(np.arange(-8.0, 8.0, 1).reshape(1,16)))
	print hl(data)

if __name__ == "__main__":
	main()

#  100: -1.8104 -1.5813 -1.3521 -1.1229 -0.8938 -0.6646 -0.4354 -0.2062  0.1000  1.1000 2.1000  3.1000  4.1000  5.1000  6.1000  7.1000
# -100: -8      -7      -6      -5      -4      -3      -2      -1       0       1      2       3       4       5       6       7
#    0: -7.9977 -6.9945 -5.9873 -4.9713 -3.9384 -2.8782 -1.7964 -0.7706  0.0525  1.0750 2.0891  3.0957  4.0984  5.0994  6.0998  7.0999
