#! /usr/bin/python

"""Tests the Debug module"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pytorchlib.debug import Debug

def main():
	xornn = nn.Sequential()
	xornn.add_module("hidden-linear", nn.Linear(2, 5))
	xornn.add_module("hidden-transfer", Debug(nn.Tanh(), "/tmp/pytorch-debug", "tanh"))
	xornn.add_module("output-logits", nn.Linear(5, 1))

	data = Variable(torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
	targets = Variable(torch.FloatTensor([[0], [1], [1], [0]]))

	opt = optim.Adamax(xornn.parameters(), lr=0.01)

	for epoch in xrange(0, 1):
		logits = xornn(data)
		loss = F.binary_cross_entropy_with_logits(logits, targets)
		loss.backward()
		opt.step()
		print "epoch=%.7f loss=%.7f" % (epoch, loss.cpu().data[0])

if __name__ == "__main__":
	main()
