#! /usr/bin/python

"""
	Debug will take a module and run the module, but store it's input and output
	and gradients and gradients in a directory
"""

import os
import torch
import torch.nn as nn

class Debug(nn.Module):
	def __init__(self, module, directory, label):
		super(Debug, self).__init__()
		self.directory = os.path.join(directory, label)
		self.index = 0
		self.label = label
		self.add_module(self.label, module)
		self.remove_handle = self.register_backward_hook(self.backward_hook)

	def forward(self, *rawdata):
		self.index += 1
		inp, = rawdata

		module = getattr(self, self.label)
		out = module(inp)

		if os.path.exists(self.directory):
			torch.save(inp.cpu(), os.path.join(self.directory, "%08d-input.torch" % (self.index)))
			torch.save(out.cpu(), os.path.join(self.directory, "%08d-output.torch" % (self.index)))

		return out

	def backward_hook(self, _, dedinput, dedoutput):
		if os.path.exists(self.directory):
			torch.save([x.cpu() for x in dedinput], os.path.join(self.directory, "%08d-dEdinput.torch" % (self.index)))
			torch.save([x.cpu() for x in dedoutput], os.path.join(self.directory, "%08d-dEdoutput.torch" % (self.index)))
		return None
