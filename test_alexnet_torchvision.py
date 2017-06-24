#! /usr/bin/python

"""This is my control. It's testing the alexnet packaged with torchvision"""

import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

#pylint: disable=too-many-locals
def classify(mlp, labels, filename):
	img = Image.open(filename)
	width, height = img.size
	size = min(width, height)
	left = (width-size) / 2
	top = (height-size) / 2
	img227 = img.crop((left, top, left+size, top+size)).resize((227, 227), Image.LANCZOS)
	img227.save("/tmp/gg.png")

	imgtensor = Variable(torch.FloatTensor(np.array(img227) * 2.0 / 255.0 - 1.0).permute(2, 0, 1).unsqueeze(0), requires_grad=False)
	results = F.softmax(mlp(imgtensor))
	probabilities, indices = results.squeeze().sort(descending=True)
	probabilities = probabilities.data.numpy().tolist()
	indices = indices.data.numpy().tolist()
	zipped = [(labels[x[0]], x[0], x[1]) for x in zip(indices, probabilities)]
	with open("/tmp/gg.json", "w") as jsonfp:
		json.dump(zipped, jsonfp, indent=4, sort_keys=True)
	for i in xrange(0, 10):
		print zipped[i]
	img.close()

def main(argv):
	alexnet = torchvision.models.alexnet(pretrained=True)
	print alexnet
	labels = []
	with open("/usr/local/caffe/data/ilsvrc12/synset_words.txt", "r") as wordsfp:
		for line in wordsfp:
			labels.append(line.strip())
	for filename in argv:
		classify(alexnet, labels, filename)

if __name__ == "__main__":
	main(sys.argv[1:])
