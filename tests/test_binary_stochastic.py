#! /usr/bin/python

"""Tests the BinaryStochastic Layer with a simple XOR"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pytorchlib.binarystochastic import BinaryStochasticLayer

def main():
	net = nn.Sequential()
	net.add_module("linear1", nn.Linear(2, 3))
	net.add_module("binary_stochastic1", nn.Tanh())
	net.add_module("linear2", nn.Linear(3, 1))
	net.add_module("binary_stochastic2", BinaryStochasticLayer(0, 1))

	data = Variable(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
	targets = Variable(torch.FloatTensor([[0], [1], [1], [0]]))

	opt = optim.Adam(net.parameters(), lr=0.001)
	loss = None
	epoch = 0

	while loss is None or loss.data.cpu()[0] > 0.01:
		epoch += 1
		opt.zero_grad()
		net.train(True)
		out = net(data)
		loss = F.mse_loss(out, targets)
		loss.backward()
		opt.step()
		net.train(False)
		out = net(data)
		loss = F.mse_loss(out, targets)

	print epoch
	print out
	print loss.data.cpu()[0]

if __name__ == "__main__":
	main()
