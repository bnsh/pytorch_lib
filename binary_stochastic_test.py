#! /usr/bin/python

import sys
import math
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from binary_stochastic import binary_stochastic

"""
So. What do we want to test?
A. The forward pass, we want to verify that the
   statistical distribution of the outputs matches what
   we expect. So, if we pass in [0.0, 0.1, 0.2, 0.3, ... 1.0]
   ten thousand times, we expect [0, 1000, 2000, 3000, 4000, ... 10000] as
   the sum of ones that we get back.
"""

def test1(sz):
	src = Variable(torch.range(0,1,0.1))
	data = Variable(src.clone().data.resize_(1,11).repeat(sz, 1))
	bs = binary_stochastic(training=True)
	rv = bs(data)
	bce = torch.nn.BCELoss(size_average=False)
	return bce(torch.mean(rv, 0), src).data[0]

def main():
	epsilon = 1e-3
	tests = 128
	sz = 16384

	# This is what we _expect_ to see.
	expectation = 0.0
	for j in xrange(1, 10):
		i = j / 10.0
		expectation -= i * math.log(i) + (1-i) * math.log(1-i)

	passes = 0
	for i in xrange(0, 128):
		rv = test1(sz)
		err = rv - expectation
		rms = math.sqrt(err*err)
		if rms < epsilon: passes += 1
		print "Forward: %d/%d Passed." % (passes, (1+i))

	# Now, let's train a neural network to encode 4 bits.

	class BitEncoder(nn.Module):
		def __init__(self, bits=4, preinitialize=False):
			super(BitEncoder, self).__init__()
			# We'll pass in a one hot vector of size 16.
			self.fc1 = nn.Linear(16, bits)
			self.fc2 = nn.Linear(bits, 16)

			if bits == 4 and preinitialize:
				self.fc1.bias.data.copy_(torch.Tensor(np.array([ -10,-10,-10,-10 ])))
				self.fc1.weight.data.copy_(torch.Tensor(np.array([
					[-10,-10,-10,-10,-10,-10,-10,-10, 20, 20, 20, 20, 20, 20, 20, 20],
					[-10,-10,-10,-10, 20, 20, 20, 20,-10,-10,-10,-10, 20, 20, 20, 20],
					[-10,-10, 20, 20,-10,-10, 20, 20,-10,-10, 20, 20,-10,-10, 20, 20],
					[-10, 20,-10, 20,-10, 20,-10, 20,-10, 20,-10, 20,-10, 20,-10, 20]
				])))
				self.fc2.bias.data.copy_(torch.Tensor(np.array([ 20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20 ])))
				self.fc2.weight.data.copy_(torch.Tensor(np.array([
					[-24,-24,-24,-24], # 0000
					[-20,-20,-20, 24], # 0001
					[-20,-20, 24,-20], # 0010
					[-20,-20, 12, 12], # 0011
					[-20, 24,-20,-20], # 0100
					[-20, 12,-20, 12], # 0101
					[-20, 12, 12,-20], # 0110
					[-20,  8,  8,  8], # 0111
					[ 24,-20,-20,-20], # 1000
					[ 12,-20,-20, 12], # 1001
					[ 12,-20, 12,-20], # 1010
					[  8,-20,  8,  8], # 1011
					[ 12, 12,-20,-20], # 1100
					[  8,  8,-20,  8], # 1101
					[  8,  8,  8,-20], # 1110
					[  6,  6,  6,  6]  # 1111
				])))
			self.encoding = None
		def forward(self, x):
			bs = binary_stochastic(training=self.training)
			x = self.fc1(x)
			x = F.sigmoid(x)
			x = self.encoding = bs(x)
			x = self.fc2(x)
			x = F.log_softmax(x)
			return x

	validation_input = Variable(torch.Tensor(np.eye(16))).cuda()
	# Now, we should just be able to pass in random one hot vectors.
	bitencoder = BitEncoder(bits=6).cuda()
	optimizer = optim.Adam(bitencoder.parameters(), lr=0.001)
	criterion = nn.KLDivLoss(size_average=False)
	epoch = 0
	while True:
		optimizer.zero_grad()
		bitencoder.train(True)
		target = np.zeros((sz,16))
		indices = np.random.randint(0, 16, (sz))
		target[np.arange(sz), indices] = 1
		src = Variable(torch.Tensor(target).cuda())
		output = bitencoder(src)
		loss = criterion(output, src)
		loss.backward()
		optimizer.step()
		epoch += 1
		if (epoch % 128) == 0:
			bitencoder.train(False)
			output = bitencoder(validation_input)
			loss = criterion(output, validation_input)
			am = np.argmax(output.data.cpu().numpy(), 0)
			accuracy = np.mean(np.equal(am, np.arange(0,16)).astype(np.float))
			print epoch, loss.data[0] / 16, accuracy

if __name__ == "__main__":
	main()
