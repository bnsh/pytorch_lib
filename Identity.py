#! /usr/bin/python

"""Identity simply copies it's inputs. WHY?! Well. Sometimes it's useful to just
   replace a layer with a "NOOP", and this basically just does that."""

import torch.nn as nn

class Identity(nn.Module):
	def forward(self, *rawdata):
		inp, = rawdata

		return inp
