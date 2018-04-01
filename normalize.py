#! /usr/bin/python

"""Normalize will adjust it's input so that it's Lp norm is k"""

import torch
import torch.nn as nn

class Normalize(nn.Module):
	# *sigh* p is just fine!
	#pylint: disable=invalid-name
	def __init__(self, p=2, dim=1, eps=1e-12):
		super(Normalize, self).__init__()
		self.p = p
		self.dim = dim
		self.eps = eps

	def forward(self, *rawdata):
		inp, = rawdata

		normalized = torch.pow(torch.sum(torch.pow(inp, self.p), self.dim), 1.0/self.p).clamp(min=self.eps).expand_as(inp)
		return inp / normalized
