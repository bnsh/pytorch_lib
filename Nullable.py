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
		indicators, data = rawdata
		# So. First, extract all the data
		nonzero = Variable(indicators.data.nonzero().squeeze(), requires_grad=False)
		portion = self.real_module(data.index_select(0, nonzero))
		size = list(portion.size())
		size = [len(indicators)] + size[1:]
		# I don't know why pylint complains about torch not having zeros
		# pylint: disable=no-member
		returnvalue = Variable(torch.zeros(size))
		self.filler(self.training, returnvalue)
		returnvalue.index_copy_(0, nonzero, portion)
		return returnvalue

