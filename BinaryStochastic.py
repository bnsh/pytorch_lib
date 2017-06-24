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

from torch.autograd.function import Function

class BinaryStochastic(Function):
	def __init__(self, training):
		super(BinaryStochastic, self).__init__()
		self.training = training

	def forward(self, *rawinp):
		inp, = rawinp
		if self.training:
			rnd = inp.clone()
			rnd.uniform_(0, 1)
			out = rnd.lt(inp).type_as(inp)
		else:
			out = inp.ge(0.5).type_as(inp)
		return out

	def backward(self, *raw_grad_output):
		grad_output = raw_grad_output
		return grad_output
