#! /usr/bin/python

"""This will test the ImageUtil class" """

import sys
import numpy as np
import torch
from PIL import Image
from pytorchlib.imageutil import ImageUtil

#pylint: disable=too-many-locals
def main(args):
	squaresz = 384

	tensors = []
	for filename in args:
		img = Image.open(filename).convert("RGB")
		width, height = img.size
		size = min(width, height)
		scaled_width = squaresz * width / size
		scaled_height = squaresz * height / size
		scaled_image = img.resize((scaled_width, scaled_height), Image.LANCZOS)
		cropped_image = scaled_image.crop(((scaled_width-squaresz)/2, (scaled_height-squaresz)/2, (scaled_width+squaresz)/2, (scaled_height+squaresz)/2))
		torchimg = torch.FloatTensor(((np.array(cropped_image).astype(np.float32) * 2.0 / 255.0) - 1)).unsqueeze(0)
		img.close()
		tensors.append(torchimg)
	inputtensor = torch.cat(tensors, 0)
	with ImageUtil() as imageutil:
		outputtensor, _, imgs = imageutil.transform(True, inputtensor)
		for i in xrange(0, len(outputtensor)):
			single_input = Image.fromarray(((imgs[i] + 1.0) * 255.0 / 2.0).clamp(0, 255).type(torch.ByteTensor).cpu().numpy())
			single_output = Image.fromarray(((outputtensor[i] + 1.0) * 255.0 / 2.0).clamp(0, 255).type(torch.ByteTensor).cpu().numpy())
			single_input.save("/tmp/image_util/input/%03d.png" % (i))
			single_output.save("/tmp/image_util/output/%03d.png" % (i))
			print i

if __name__ == "__main__":
	main(sys.argv[1:])
