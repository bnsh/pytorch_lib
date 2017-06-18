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

class nullable(nn.Module):
	def __init__(self, real_module, filler):
		super(nullable, self).__init__()
		self.real_module = real_module
		self.filler = filler

	def forward(self, pair):
		indicators, data = pair
		# So. First, extract all the data
		nz = Variable(torch.nonzero(indicators.data).squeeze(), requires_grad=False)
		portion = self.real_module(data.index_select(0,nz))
		sz = list(portion.size())
		sz = [len(indicators)] + sz[1:]
		rv = Variable(torch.zeros(sz))
		self.filler(self.training, rv)
		rv.index_copy_(0, nz, portion)
		return rv

