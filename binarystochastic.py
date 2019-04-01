#! /usr/bin/python

"""BinaryStochastic interprets it's input as a probability
between [0,1] and
 * with probability p, outputs a 1, and
 * with probability (1-p) it outputs a 0.

Practically, this is accomplished simply by taking a random uniform
the size of the input and seeing if the random uniform is less than
the input. If true, we output a 1, if false we output a 0.

The limits are _not_ enforced! If you pass a -2 to it, it will
_always_ output 0 and if you pass a 2 to it, it will _always_ output
a 1! Caveat emptor and all that.
"""

import torch.nn as nn
from torch.autograd.function import Function

class BinaryStochasticRaw(Function):
	#pylint: disable=arguments-differ
	@staticmethod
	def forward(ctx, inp, training):
		if training:
			rnd = inp.clone()
			rnd.uniform_(0, 1)
			out = rnd.lt(inp).type_as(inp)
		else:
			out = inp.ge(0.5).type_as(inp)
		return out
	#pylint: enable=arguments-differ

	@staticmethod
	def backward(ctx, *grad_outputs):
		returned_grad_output = grad_outputs
		return returned_grad_output[0], None

class BinaryStochasticRawLayer(nn.Module):
	def __init__(self, loval=0, hival=1):
		super(BinaryStochasticRawLayer, self).__init__()
		self.loval = loval
		self.hival = hival

	def forward(self, *args):
		tensor, = args
		binary_stochastic = BinaryStochasticRaw.apply
		return self.loval + binary_stochastic(tensor, self.training) * (self.hival-self.loval)

class BinaryStochasticLayer(nn.Sequential):
	def __init__(self, loval=0, hival=1):
		super(BinaryStochasticLayer, self).__init__()
		self.add_module("sigmoid", nn.Sigmoid())
		self.add_module("binary_stochastic", BinaryStochasticRawLayer(loval, hival))
