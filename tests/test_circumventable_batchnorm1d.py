#! /usr/bin/python

"""This program tests the CircumventableBatchNorm1D module. (I'm not actually using it
   anymore, but still..)"""

import torch
from torch.autograd import Variable

from pytorchlib.CircumventableBatchNorm1D import CircumventableBatchNorm1D

def test_disabled():
	data = Variable(torch.randn(20, 4))

	cbn = CircumventableBatchNorm1D(4)
	before = cbn.running_mean.clone()
	cbn.stop_running_stats()
	_ = cbn(data)
	after = cbn.running_mean.clone()
	assert torch.pow(before-after, 2).sum() == 0

def test_enabled():
	data = Variable(torch.randn(20, 4))

	cbn = CircumventableBatchNorm1D(4)
	before = cbn.running_mean.clone()
	cbn.start_running_stats()
	_ = cbn(data)
	after = cbn.running_mean.clone()
	assert torch.pow(before-after, 2).sum() > 0

def main():
	test_disabled()
	test_enabled()



if __name__ == "__main__":
	main()
