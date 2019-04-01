#! /usr/bin/env python3

"""This class _should_ take a model, and return a container model that will do
   Stochastic Delta Rule on the model transparently. This is from:
   https://arxiv.org/abs/1808.03578
"""

import torch
import torch.nn as nn
import numpy as np

#pylint: disable=too-many-instance-attributes
class SDR(nn.Module):
	#pylint: disable=too-many-arguments
	def __init__(self, model, alpha=0.25, beta=0.2, zeta=0.99, n_blocks=2, zeta_ratio=0.9):
		super(SDR, self).__init__()
		self.model = model
		self.alpha = alpha
		self.beta = beta
		self.zeta_orig = zeta
		self.zeta = zeta
		self.n_blocks = n_blocks
		self.zeta_ratio = zeta_ratio
		self.sds = []
		self.data_swap = []
	#pylint: enable=too-many-arguments

	#pylint: disable=too-many-locals
	def forward(self, *args):
		inp = args[0]
		# So, first, we need to update the standard deviations.
		if self.training:
			length = len(list(self.model.parameters()))
			for paramidx, param in enumerate(self.model.parameters()):
				if (1+paramidx) > len(self.sds):
					# This is right from https://github.com/noahfl/sdr-densenet-pytorch
					sd_min = 0.0
					sd_max = np.sqrt(2. / np.product(param.shape)) * 0.5

					res = torch.randn(param.data.shape)
					mxx = torch.max(res)
					mnn = torch.min(res)

					init = ((sd_max-sd_min) / (mxx-mnn)).float() * (res - mnn)
					if param.data.is_cuda:
						init = init.cuda(param.data.get_device())

					self.sds.append(init)
				else:
					# So, we need to divide the length parameters into
					# n_blocks.
					frac = int(self.n_blocks * (1-float(paramidx)/length))
					zeta_ = self.zeta * self.zeta_ratio ** frac
					self.sds[paramidx] = zeta_ * (torch.abs(self.beta * param.grad) + self.sds[paramidx])

			self.data_swap.clear()
			for paramidx, param in enumerate(self.model.parameters()):
				self.data_swap.append(param.data)
				param.data = torch.distributions.Normal(param.data, self.sds[paramidx]).sample()

		retval = self.model(inp)

		if self.training:
			for param, backup in zip(self.model.parameters(), self.data_swap):
				param.data = backup
		return retval
	#pylint: enable=too-many-locals

	def update_zeta(self, epoch):
		self.zeta = min(0.01, self.zeta_orig * np.power(np.e, -(0.1 * epoch)))
#pylint: disable=too-many-instance-attributes

def main():
	data = torch.ones((17, 5)).cpu()
	lin = nn.Linear(5, 3).cpu()
	torch.nn.init.eye_(lin.weight)
	torch.nn.init.constant_(lin.bias, 0)
	sdrlin = SDR(lin)

	print(sdrlin(data))

if __name__ == "__main__":
	main()
