#! /usr/bin/python

"""This class performs image transformations useful for a convnet"""

import random
import torch
from . import imageutil_cext

class ImageUtil(object):
	def __init__(self, insz=360, outsz=227):
		#pylint: disable=no-member
		self.iuptr = imageutil_cext.ImageUtil()
		self.insz = insz
		self.outsz = outsz

	def __enter__(self):
		#pylint: disable=no-member
		if self.iuptr is not None:
			imageutil_cext.ImageUtil_destroy(self.iuptr)
		self.iuptr = imageutil_cext.ImageUtil()
		return self

	def __exit__(self, _, value, traceback):
		#pylint: disable=no-member
		imageutil_cext.ImageUtil_destroy(self.iuptr)
		self.iuptr = None

	def transform(self, randomize, images):
		return_value = None
		if randomize:
			return_value = self.transform_random(images)
		else:
			return_value = self.transform_center(images)
		return return_value


	#pylint: disable=too-many-locals
	def transform_random(self, images):
		assert self.iuptr is not None
# We need to grab random 384x384 crops of images.
		inputbatch = torch.zeros(len(images), self.insz, self.insz, 3).cuda()
		outputbatch = torch.zeros(len(images), self.outsz, self.outsz, 3).cuda()
		conversion_matrix = torch.eye(3).resize_(1, 3, 3).repeat(len(images), 1, 1).cuda()

		for i in xrange(0, len(images)):
			assert images[i].size()[0] >= self.insz
			assert images[i].size()[1] >= self.insz
			thresh = 0.5
			mnval = 0-thresh
			mxval = 1+thresh
			randomy = min(max(mnval + random.random() * (mxval-mnval), 0), 1)
			assert (0 <= randomy) and (randomy <= 1)
			scaledy = int(randomy * (images[i].size()[0]-self.insz))

			randomx = min(max(mnval + random.random() * (mxval-mnval), 0), 1)
			assert (0 <= randomx) and (randomx <= 1)
			scaledx = int(randomx * (images[i].size()[1]-self.insz))
			conversion_matrix[i, 0, 2] = -scaledy
			conversion_matrix[i, 1, 2] = -scaledx
			inputbatch[i, :, :, :].copy_(images[i][scaledy:(scaledy+self.insz), scaledx:(scaledx+self.insz), :])


		#pylint: disable=no-member
		imageutil_cext.ImageUtil_transform(self.iuptr, True, inputbatch, outputbatch, conversion_matrix)
		return outputbatch, conversion_matrix, images

	#pylint: disable=too-many-locals
	def transform_tile(self, orig_images):
		assert self.iuptr is not None
		subimages = []
		images = []
		conversions = []
		for qidx in xrange(0, len(orig_images)):
			image = orig_images[qidx]
			height = image.size(0)
			width = image.size(1)
			squaresw = (width / self.insz) + 1
			squaresh = (height / self.insz) + 1

			for i in xrange(0, squaresh):
				top = i * (height - self.insz) / (squaresh-1)
				for j in xrange(0, squaresw):
					left = j * (width - self.insz) / (squaresw-1)
					images.append(image)
					subimages.append(image[top:(top+self.insz), left:(left+self.insz), :])
					cmatrix = torch.eye(3)
					cmatrix[0, 2] = -top
					cmatrix[1, 2] = -left
					conversions.append(cmatrix)

		inputbatch = torch.stack(subimages, 0).float().cuda()
		outputbatch = torch.zeros(len(subimages), self.outsz, self.outsz, 3).cuda()
		conversion_matrix = torch.stack(conversions, 0).cuda()

		#pylint: disable=no-member
		imageutil_cext.ImageUtil_transform(self.iuptr, False, inputbatch, outputbatch, conversion_matrix)
		del inputbatch

		return outputbatch, conversion_matrix, images

	#pylint: disable=too-many-locals
	def transform_center(self, orig_images):
		assert self.iuptr is not None
		subimages = []
		images = []
		conversions = []
		for qidx in xrange(0, len(orig_images)):
			image = orig_images[qidx]
			height = image.size(0)
			width = image.size(1)

			top = int(0.05 * height)
			left = int(0.05 * width)
			images.append(image)
			subimages.append(image[top:(int(top+0.9*height)), left:(int(left+0.9*width)), :])
			cmatrix = torch.eye(3)
			cmatrix[0, 2] = -top
			cmatrix[1, 2] = -left
			conversions.append(cmatrix)

		inputbatch = torch.stack(subimages, 0).float().cuda() # We need to convert these to floats
		outputbatch = torch.zeros(len(subimages), self.outsz, self.outsz, 3).cuda()
		conversion_matrix = torch.stack(conversions, 0).cuda()

		#pylint: disable=no-member
		imageutil_cext.ImageUtil_transform(self.iuptr, False, inputbatch, outputbatch, conversion_matrix)
		del inputbatch

		return outputbatch, conversion_matrix, images
