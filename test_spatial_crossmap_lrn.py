#! /usr/bin/python

"""This tests SpatialCrossMapLRN.
   I have verified that my numpy version computes the same values as
   the LuaTorch SpatialCrossMapLRN."""

import sys
import math
import numpy as np
import torch
from torch.autograd import Variable
from SpatialCrossMapLRN import SpatialCrossMapLRN

def my_scmlrn(size, alpha, beta, k, inp):
	output = np.zeros(inp.shape)
	for feature_idx in xrange(0, inp.shape[1]):
		numerator = inp[:, feature_idx, :, :]
		lower_bound = int(max(0, feature_idx - math.floor(size/2.0)))
		upper_bound = int(min(inp.shape[1]-1, feature_idx + math.floor(size/2.0)))
		denominator = np.power((k + (alpha/size) * np.power(inp[:, np.arange(lower_bound, upper_bound+1), ::], 2)).sum(1), beta)
		output[:, feature_idx, :, :] = numerator / denominator
	return output

def main(_):
	size = 5
	alpha = 0.0001
	beta = 0.75
	k = 1

	minibatches = 10
	features = 10
	height = 10
	width = 10
	inp = np.arange(0, minibatches*features*height*width).reshape(minibatches, features, height, width)

	scmlrn = SpatialCrossMapLRN(size, alpha, beta, k)
	np_out = my_scmlrn(size, alpha, beta, k, inp)
	th_out = scmlrn(Variable(torch.DoubleTensor(inp))).data.numpy()
	print np.power((np_out - th_out), 2).max()

if __name__ == "__main__":
	main(sys.argv[1:])
