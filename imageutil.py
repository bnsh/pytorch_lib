#! /usr/bin/python

"""This class performs image transformations useful for a convnet"""

import numpy as np
import torch
import cv2
from albumentations import (
	Compose,
	OneOf,
	IAAAdditiveGaussianNoise,
	GaussNoise,
	MotionBlur,
	MedianBlur,
	Blur,
	Rotate,
	ShiftScaleRotate,
	OpticalDistortion,
	GridDistortion,
	IAAPiecewiseAffine,
	CLAHE,
	IAASharpen,
	IAAEmboss,
	RandomContrast,
	RandomBrightness,
	RandomGamma,
	HueSaturationValue,
	Resize
)

class ImageUtil(object):
	def __init__(self, insz=360, outsz=227):
		self.insz = insz
		self.outsz = outsz

	def __enter__(self):
		return self

	def __exit__(self, _, value, traceback):
		pass

	def transform(self, randomize, images):
		return_value = None
		if randomize:
			return_value = self._transform_random(images)
		else:
			return_value = self._transform_center(images)
		return return_value


	def _transform_generic(self, images, prob):
		"""So, we assume these images are batched torch values, scaled from -1..1
			We need to convert them to numpy arrays.
		"""

		batchsz, height, width, channels = images.shape
		assert height == width, "We assume squares as inputs."
		assert channels == 3, "We assume RGB images."
		assert -1.0 <= images.min() and images.max() <= 1.0

		# make these the images that albumentation requires.
		npimages = ((images.cpu().numpy() + 1.0) * 255.0 / 2.0).astype(np.uint8)
		outputs = np.zeros((batchsz, self.outsz, self.outsz, channels))

		ops = Compose( \
			[ \
				Compose( \
					[ \
						OneOf([ \
							IAAAdditiveGaussianNoise(p=1.0), \
							GaussNoise(p=1.0), \
						], p=0.5), \
						OneOf([ \
							MotionBlur(p=1.0), \
							MedianBlur(blur_limit=3, p=1.0), \
							Blur(blur_limit=3, p=1.0), \
						], p=0.5), \
						RandomGamma(p=0.5), \
						Rotate(limit=45, interpolation=cv2.INTER_CUBIC, p=0.5), \
						ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, interpolation=cv2.INTER_CUBIC, p=0.5), \
						OneOf([ \
							OpticalDistortion(interpolation=cv2.INTER_CUBIC, p=1.0), \
							GridDistortion(interpolation=cv2.INTER_CUBIC, p=1.0), \
							IAAPiecewiseAffine(p=1.0), \
						], p=0.5), \
						OneOf([ \
							CLAHE(clip_limit=2, p=1.0), \
							IAASharpen(p=1.0), \
							IAAEmboss(p=1.0), \
							RandomContrast(p=1.0), \
							RandomBrightness(p=1.0), \
						], p=0.5), \
						HueSaturationValue(p=0.5), \
					], \
					p=prob \
				) \
			], \
			postprocessing_transforms=[Resize(self.outsz, self.outsz, interpolation=cv2.INTER_CUBIC)], \
			p=1.0 \
		)

		# So, the output of ops, should be a dictionary containing an image
		for idx in range(0, batchsz):
			vvv = ops(image=npimages[idx])["image"]
			outputs[idx] = vvv

		return torch.Tensor(outputs).type_as(images)

	def _transform_random(self, images):
		return self._transform_generic(images, prob=0.5)

	def _transform_center(self, images):
		return self._transform_generic(images, prob=0.0)
