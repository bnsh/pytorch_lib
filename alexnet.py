#! /usr/bin/python

"""AlexNet is basically my implementation of AlexNet
with some minor changes."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from .spatialcrossmaplrn import SpatialCrossMapLRN

def conv1(dropout, transfer):
	"""conv1 takes the image as input and outputs a feature map of size 96"""
	mlp = nn.Sequential()
	mlp.add_module("conv2d", nn.Conv2d(3, 96, (11, 11), (4, 4), padding=0))
	mlp.add_module("transfer", transfer())
	mlp.add_module("lrn", SpatialCrossMapLRN(5, 0.0001, 0.75))
	mlp.add_module("max_pool", nn.MaxPool2d((3, 3), (2, 2)))
	mlp.add_module("dropout", nn.Dropout2d(dropout))
	return mlp

def conv2(dropout, transfer):
	"""conv2 takes the 96 features, splits it into 2 48 features and runs over
each of those to produce 2 128 features and then joins them both again
as 256 features."""
	class AlexNetConv2(nn.Module):
		def __init__(self, dropout, transfer):
			super(AlexNetConv2, self).__init__()
			self.dropout = dropout
			self.transfer = transfer
			self.conv1 = nn.Conv2d(48, 128, (5, 5), (1, 1), (2, 2))
			self.conv2 = nn.Conv2d(48, 128, (5, 5), (1, 1), (2, 2))
			self.lrn = SpatialCrossMapLRN(5, 0.0001, 0.75)
			self.maxpooling = nn.MaxPool2d((3, 3), (2, 2))
			self.dropout = nn.Dropout2d(dropout)
			self.transfer = transfer()

		def forward(self, *args):
			data, = args

			indices_0_48 = Variable(torch.arange(0, 48).long(), requires_grad=False)
			indices_48_96 = Variable(torch.arange(48, 96).long(), requires_grad=False)
			if data.is_cuda:
				indices_0_48 = indices_0_48.cuda()
				indices_48_96 = indices_48_96.cuda()
			path1 = self.conv1(data.index_select(1, indices_0_48))
			path2 = self.conv2(data.index_select(1, indices_48_96))

			full = torch.cat((path1, path2), 1) # index 1 are the features
			return_value = self.transfer(full)
			return_value = self.lrn(return_value)
			return_value = self.maxpooling(return_value)
			return_value = self.dropout(return_value)

			return return_value

	return AlexNetConv2(dropout, transfer)

def conv3(dropout, transfer):
	mlp = nn.Sequential()
	mlp.add_module("conv2d", nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)))
	mlp.add_module("transfer", transfer())
	mlp.add_module("dropout", nn.Dropout2d(dropout))
	return mlp

def conv4(dropout, transfer):
	class AlexNetConv4(nn.Module):
		def __init__(self, dropout, transfer):
			super(AlexNetConv4, self).__init__()
			self.dropout = dropout
			self.transfer = transfer
			self.conv1 = nn.Conv2d(192, 192, (3, 3), (1, 1), (1, 1))
			self.conv2 = nn.Conv2d(192, 192, (3, 3), (1, 1), (1, 1))
			self.dropout = nn.Dropout2d(dropout)
			self.transfer = transfer()

		def forward(self, *args):
			data, = args

			indices_0_192 = Variable(torch.arange(0, 192).long(), requires_grad=False)
			indices_192_384 = Variable(torch.arange(192, 384).long(), requires_grad=False)
			if data.is_cuda:
				indices_0_192 = indices_0_192.cuda()
				indices_192_384 = indices_192_384.cuda()
			path1 = self.conv1(data.index_select(1, indices_0_192))
			path2 = self.conv2(data.index_select(1, indices_192_384))

			full = torch.cat((path1, path2), 1) # index 1 are the features
			return_value = self.transfer(full)
			return_value = self.dropout(return_value)

			return return_value

	return AlexNetConv4(dropout, transfer)

def conv5(dropout, transfer):
	class AlexNetConv5(nn.Module):
		def __init__(self, dropout, transfer):
			super(AlexNetConv5, self).__init__()
			self.dropout = dropout
			self.transfer = transfer
			self.conv1 = nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1))
			self.conv2 = nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1))
			self.maxpooling = nn.MaxPool2d((3, 3), (2, 2))
			self.dropout = nn.Dropout2d(dropout)
			self.transfer = transfer()

		def forward(self, *args):
			data, = args

			indices_0_192 = Variable(torch.arange(0, 192).long(), requires_grad=False)
			indices_192_384 = Variable(torch.arange(192, 384).long(), requires_grad=False)
			if data.is_cuda:
				indices_0_192 = indices_0_192.cuda()
				indices_192_384 = indices_192_384.cuda()
			path1 = self.conv1(data.index_select(1, indices_0_192))
			path2 = self.conv2(data.index_select(1, indices_192_384))

			full = torch.cat((path1, path2), 1) # index 1 are the features
			return_value = self.transfer(full)
			return_value = self.maxpooling(return_value)
			return_value = self.dropout(return_value)

			return return_value

	return AlexNetConv5(dropout, transfer)

def fc6(size, dropout, transfer):
	class AlexNetFC6(nn.Module):
		def __init__(self, size, dropout, transfer):
			super(AlexNetFC6, self).__init__()
			self.linear = nn.Linear(256*6*6, size)
			self.dropout = nn.Dropout(dropout)
			self.transfer = transfer()

		def forward(self, *args):
			data, = args
			return_value = data.resize(data.size(0), data.size(1)*data.size(2)*data.size(3))
			return_value = self.linear(return_value)
			return_value = self.transfer(return_value)
			return_value = self.dropout(return_value)

			return return_value
	return AlexNetFC6(size, dropout, transfer)

def fc7(size, dropout, transfer):
	mlp = nn.Sequential()
	mlp.add_module("linear", nn.Linear(size, 4096))
	mlp.add_module("transfer", transfer())
	mlp.add_module("dropout", nn.Dropout(dropout))

	return mlp

def fc8(size):
	mlp = nn.Sequential()
	mlp.add_module("linear", nn.Linear(4096, size))

	return mlp

def defaults(params, default):
	if params is None:
		params = {}
	for key, value in default.iteritems():
		if key in params:
			params[key] = params[key]
		else:
			params[key] = value
	return params

class AlexNet(nn.Sequential):
	def __init__(self, params):
		super(AlexNet, self).__init__()
		if params is None:
			params = {}
		for k in ("general", "bottleneck", "output"):
			if k not in params:
				params[k] = {}

		general_params = defaults(params["general"], {"dropout": 0.5, "transfer": nn.ReLU})
		bottleneck_params = defaults(params["bottleneck"], {"sz": 4096, "dropout": 0.5, "transfer": nn.ReLU})
		output_params = defaults(params["output"], {"sz": 1000, "dropout": 0.5, "transfer": nn.ReLU})

		self.add_module("conv1", conv1(general_params["dropout"], general_params["transfer"]))
		self.add_module("conv2", conv2(general_params["dropout"], general_params["transfer"]))
		self.add_module("conv3", conv3(general_params["dropout"], general_params["transfer"]))
		self.add_module("conv4", conv4(general_params["dropout"], general_params["transfer"]))
		self.add_module("conv5", conv5(general_params["dropout"], general_params["transfer"]))
		self.add_module("fc6", fc6(bottleneck_params["sz"], bottleneck_params["dropout"], bottleneck_params["transfer"]))
		self.add_module("fc7", fc7(bottleneck_params["sz"], general_params["dropout"], general_params["transfer"]))
		self.add_module("fc8", fc8(output_params["sz"]))
