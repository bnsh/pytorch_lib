#! /usr/bin/python

"""Xavier does Xavier initialization given a _module_"""

import torch.nn.init as init

def xavier(params):
	for param in params.parameters():
		if param.data.ndimension() == 2:
			init.xavier_uniform(param.data)
		else:
			param.data.fill_(1.0)
	return params

