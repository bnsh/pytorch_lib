#! /usr/bin/python

"""This just reads a random_source as implemented in ../torch7-libv2/random_source.lua"""

import torch

def random_source(seed, rsize, generator=torch.randn):
	torch.manual_seed(seed)
	random_src = generator(rsize)
	pos = [0]
	def grab(size):
		returnvalue = torch.FloatTensor(size).zero_()
		rpos = 0

		while rpos < size:
			grabsz = size
			if pos[0] + grabsz >= random_src.size(0):
				grabsz = random_src.size(0) - pos[0]

			returnvalue.index_copy_(0, torch.arange(rpos, rpos+grabsz).long(), random_src[pos[0]:(pos[0]+grabsz)])
			pos[0] = (pos[0] + grabsz) % random_src.size(0)
			rpos += grabsz

		return returnvalue
	return grab
