#! /usr/bin/python

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def xavier(params):
	for p in params.parameters():
		if p.data.ndimension() == 2:
			init.xavier_uniform(p.data)
		else:
			p.data.fill_(1.0)
	return params

