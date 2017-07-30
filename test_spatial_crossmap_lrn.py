#! /usr/bin/python

"""This tests SpatialCrossMapLRN.
   I have verified that my numpy version computes the same values as
   the LuaTorch SpatialCrossMapLRN.
   2017-07-30 - Actually, while _close_ this _doesn't_ unfortunately match the
                values from the torch version. I'm giving up tho, because it seems
		like I can't _exactly_ duplicate the torch7 version while keeping with
		the "Variable" concept. (Basically, the old one didn't have to worry
		about Variable, because it did the derivative directly, so it could
		just do things like copy values and replace values etc, at will.)
		My way while not preserving the exact values, doesn't require me
		to implement backward as well.
"""

import sys
import math
import numpy as np
import torch
from torch.autograd import Variable
from pytorchlib.random_source import random_source
from pytorchlib.SpatialCrossMapLRN import SpatialCrossMapLRN

def simple_json(filehandle, indentlevel, nparray):
	indentstr = "\t" * indentlevel
	if nparray.ndim == 1:
		filehandle.write("%s[" % (indentstr))
		for idx, val in enumerate(nparray.tolist()):
			if idx > 0:
				filehandle.write(", ")
			filehandle.write("%.7f" % (val))
		filehandle.write("]")
	else:
		filehandle.write("%s[" % (indentstr))
		for idx, val in enumerate(nparray):
			if idx > 0:
				filehandle.write(", ")
			filehandle.write("\n")
			simple_json(filehandle, (1+indentlevel), val)
		filehandle.write("\n%s]" % (indentstr))

def my_scmlrn(size, alpha, beta, k, inp):
	output = np.zeros(inp.shape)
	for feature_idx in xrange(0, inp.shape[1]):
		numerator = inp[:, feature_idx, :, :]
		lower_bound = int(max(0, feature_idx - math.floor(size/2.0)))
		upper_bound = int(min(inp.shape[1]-1, feature_idx + math.floor(size/2.0)))
		denominator = np.power((k + (alpha/size) * np.power(inp[:, np.arange(lower_bound, upper_bound+1), ::], 2).sum(1)), beta)
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

	grab = random_source(12345, 16777216)
	inp = grab(minibatches*features*height*width).resize_(minibatches, features, height, width)

	scmlrn = SpatialCrossMapLRN(size, alpha, beta, k)
	np_out = my_scmlrn(size, alpha, beta, k, inp.numpy())
	sc_out = scmlrn(Variable(inp)).data.numpy()
	print np.power((np_out - sc_out), 2).max()

	with open("/tmp/np.json", "w") as filehandle:
		simple_json(filehandle, 0, np_out)

	with open("/tmp/scm.json", "w") as filehandle:
		simple_json(filehandle, 0, sc_out)

if __name__ == "__main__":
	main(sys.argv[1:])
