#! /usr/bin/python

"""This is basically BatchNorm1d, but, I want to be able to _sometimes_ preserve
   the running_var and running_mean. I'm going to try doing this by always setting
   training _off_ before running forward _if_ it's been disabled.
   sometimes I need this if I _only_ want to train a part of the network.
"""

import torch.nn as nn
import torch.nn.functional as F

class CircumventableBatchNorm1D(nn.BatchNorm1d):
	#pylint: disable=too-many-arguments
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, start_disabled=False):
		super(CircumventableBatchNorm1D, self).__init__(num_features, eps, momentum, affine)
		self.disabled = start_disabled
	#pylint: enable=too-many-arguments

	# *sigh* because BatchNorm1d has forward(self, input).
	# I could either disable arguments-differ, or disable redefined-builtin
	# I'm opting for arguments-differ.
	#pylint: disable=arguments-differ
	def forward(self, inp):
		# If training is true, running_var gets updated.
		# if disabled is true  running_var should _not_ be updated.
		# if training is true and disabled is false running_var gets updated
		return F.batch_norm(inp, self.running_mean, self.running_var, self.weight, self.bias, self.training and not self.disabled, self.momentum, self.eps)
	#pylint: enable=arguments-differ

	def stop_running_stats(self):
		self.disabled = True

	def start_running_stats(self):
		self.disabled = False
