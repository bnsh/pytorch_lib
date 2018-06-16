#! /usr/bin/python

"""Nullable allows a neural network to handle "null" inputs."""

import torch.nn as nn
from torch.autograd import Variable

class Nullable(nn.Module):
	def __init__(self, module, fillmethod):
		super(Nullable, self).__init__()
		self.module = module
		self.fillmethod = fillmethod

	def forward(self, *args):
		inp, = args
		indicator = inp["indicator"]
		data = inp["data"]
		output = self.module(data)
		random = Variable(self.fillmethod(self.training, output.shape).type_as(data.detach()))
		return output * indicator + (1-indicator) * random
