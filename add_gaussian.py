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

class add_gaussian(nn.Module):
	def __init__(self, mean, sigma):
		super(add_gaussian, self).__init__()
		self.mean = mean
		self.sigma = sigma

	def forward(self, data):
		if self.training:
			noise = Variable(data.data.clone().normal_(self.mean, self.sigma), requires_grad=False)
			return data + noise
		else:
			return data

def main():
	ag = add_gaussian(0, 1)
	ag.train(True)
	data = Variable(torch.Tensor(np.arange(-8.0, 8.0, 1).reshape(1,16)))
	print ag(data)

if __name__ == "__main__":
	main()

