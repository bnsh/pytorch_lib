#! /usr/bin/python

"""This tests the Normalize method."""

import torch
from torch.autograd import Variable
from Normalize import Normalize

def floatrange(low, high, increment):
	curr = low
	while curr < high:
		yield curr
		curr += increment

# p and lp are widely recognizable in this context.
#pylint: disable=invalid-name
def main():
	for p in floatrange(0.5, 10, 0.5):
		lpnorm = Normalize(p=p)
		data = Variable(torch.randn(10, 10))
		lp = lpnorm(data)
		lpverify = torch.pow(lp, p).sum(1)
		assert 0.9999 <= lpverify and lpverify <= 1.0001

if __name__ == "__main__":
	main()
