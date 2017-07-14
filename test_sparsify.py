#! /usr/bin/python

import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Sparsify import Sparsify

def main():
	sparsify = Sparsify(k=2, replacement=-1)
	data = Variable(torch.arange(0,10).resize_(1,10))
	print data
	print sparsify(data)

if __name__ == "__main__":
	main()
