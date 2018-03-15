#! /usr/bin/python

"""In principle this should be taken care of by
	http://pytorch.org/docs/0.3.1/nn.html#weight-norm
   But, it's so slow (nearly 3 times as slow when running it.) So..
   I figured, why not try writing my own.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init

class WeightNormalizedLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super(WeightNormalizedLinear, self).__init__()

		self.register_parameter('weight_v', Parameter(torch.FloatTensor(out_features, in_features)))
		self.register_parameter('weight_g', Parameter(torch.FloatTensor(out_features, 1)))
		self.register_parameter('bias', Parameter(torch.FloatTensor(out_features, 1)))

		# Zero initialize the bias
		init.orthogonal(self.weight_v)
		init.constant(self.weight_g, 1.0)
		init.constant(self.bias, 0.0)

	def forward(self, *args):
		inp, = args

		return (torch.matmul(self.weight_g * self.weight_v / torch.norm(self.weight_v, p=2, dim=0, keepdim=True), inp.transpose(1, 0)) + self.bias).transpose(1, 0)
