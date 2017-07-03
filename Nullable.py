#! /usr/bin/python

"""Nullable allows a neural network to handle "null" inputs."""

import torch
import torch.nn as nn
from torch.autograd import Variable

class Nullable(nn.Module):
	def __init__(self, real_module, filler):
		super(Nullable, self).__init__()
		self.real_module = real_module
		self.filler = filler

	def forward(self, *rawdata):
		indicators = rawdata[0]["indicators"]
		data = rawdata[0]["data"]
		assert (indicators.is_cuda and data.is_cuda) or not(indicators.is_cuda or data.is_cuda)
		# So. First, extract all the data
		nonzero = Variable(indicators.data.squeeze().nonzero().squeeze(), requires_grad=False)
		if nonzero.dim() > 0:
			output = self.real_module(data.index_select(0, nonzero))
			size = list(output.size())
			size[0] = len(indicators)

			returnvalue = Variable(torch.zeros(size)).type_as(data)
			self.filler(self.training, returnvalue)
			returnvalue.index_copy_(0, nonzero, output)
		else:
			# Well. We don't have _any_ inputs. Let's send a
			# dummy sample through, and see what the size _would_
			# have been.
			dummy_size = list(data.size())
			dummy_size[0] = 1 # Let's just send one in
			dummy_data = Variable(torch.zeros(dummy_size), requires_grad=False).type_as(data)
			dummy_output = self.real_module(dummy_data)
			size = list(dummy_output.size())
			size[0] = len(indicators)
			returnvalue = Variable(torch.zeros(size)).type_as(data)
			self.filler(self.training, returnvalue)

		return returnvalue
