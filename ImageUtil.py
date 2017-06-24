#! /usr/bin/python

"""This class performs image transformations useful for a convnet"""

import random
import torch
import ImageUtil_cext

class ImageUtil(object):
	def __init__(self):
		#pylint: disable=no-member
		self.iuptr = ImageUtil_cext.ImageUtil()

	def __enter__(self):
		#pylint: disable=no-member
		if self.iuptr is not None:
			ImageUtil_cext.ImageUtil_destroy(self.iuptr)
		self.iuptr = ImageUtil_cext.ImageUtil()
		return self

	def __exit__(self, _, value, traceback):
		#pylint: disable=no-member
		ImageUtil_cext.ImageUtil_destroy(self.iuptr)
		self.iuptr = None

	def transform(self, randomize, images):
		return_value = None
		if randomize:
			return_value = self.transform_random(images)
		else:
			return_value = self.transform_tile(images)
		return return_value


	#pylint: disable=too-many-locals
	def transform_random(self, images):
		assert self.iuptr is not None
# We need to grab random 384x384 crops of images.
		insz = 256
		outsz = 224
		inputbatch = torch.zeros(len(images), insz, insz, 3).cuda()
		outputbatch = torch.zeros(len(images), outsz, outsz, 3).cuda()
		conversion_matrix = torch.eye(3).resize_(1, 3, 3).repeat(len(images), 1, 1).cuda()

		for i in xrange(0, len(images)):
			assert images[i].size()[0] >= insz
			assert images[i].size()[1] >= insz
			thresh = 0.5
			mnval = 0-thresh
			mxval = 1+thresh
			randomy = min(max(mnval + random.random() * (mxval-mnval), 0), 1)
			assert (0 <= randomy) and (randomy <= 1)
			scaledy = int(randomy * (images[i].size()[0]-insz))

			randomx = min(max(mnval + random.random() * (mxval-mnval), 0), 1)
			assert (0 <= randomx) and (randomx <= 1)
			scaledx = int(randomx * (images[i].size()[1]-insz))
			conversion_matrix[i, 0, 2] = -scaledy
			conversion_matrix[i, 1, 2] = -scaledx
			inputbatch[i, :, :, :].copy_(images[i][scaledy:(scaledy+insz), scaledx:(scaledx+insz), :])


		#pylint: disable=no-member
		ImageUtil_cext.ImageUtil_transform(self.iuptr, True, inputbatch, outputbatch, conversion_matrix)
		return outputbatch, conversion_matrix, images

	#pylint: disable=too-many-locals
	def transform_tile(self, orig_images):
		assert self.iuptr is not None
		subimages = []
		images = []
		conversions = []
		insz = 256
		outsz = 224
		for qidx in xrange(0, len(orig_images)):
			image = orig_images[qidx]
			height = image.size(0)
			width = image.size(1)
			squaresw = (width / insz) + 1
			squaresh = (height / insz) + 1

			for i in xrange(0, squaresh):
				top = i * (height - insz) / (squaresh-1)
				for j in xrange(0, squaresw):
					left = j * (width - insz) / (squaresw-1)
					images.append(image)
					subimages.append(image[top:(top+insz), left:(left+insz), :])
					cmatrix = torch.eye(3)
					cmatrix[0, 2] = -top
					cmatrix[1, 2] = -left
					conversions.append(cmatrix)

		inputbatch = torch.stack(subimages, 0).cuda()
		outputbatch = torch.zeros(len(subimages), outsz, outsz, 3).cuda()
		conversion_matrix = torch.stack(conversions, 0).cuda()

		#pylint: disable=no-member
		ImageUtil_cext.ImageUtil_transform(self.iuptr, False, inputbatch, outputbatch, conversion_matrix)
		del inputbatch

		return outputbatch, conversion_matrix, images
