#! /usr/bin/python

"""This will do some basic tests of ResNet"""

import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pytorchlib.resnet import resnet152

#pylint: disable=too-many-locals
def classify(mlp, labels, filename):
	img = Image.open(filename)
	width, height = img.size
	size = min(width, height)
	left = (width-size) / 2
	top = (height-size) / 2
	# from https://pytorch.org/docs/stable/torchvision/models.html
	# All pre-trained models expect input images normalized in the same way,
	# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where
	# H and W are expected to be at least 224. The images have to be loaded
	# in to a range of [0, 1] and then normalized using
	# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
	# You can use the following transform to normalize:
	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	img224 = img.crop((left, top, left+size, top+size)).resize((224, 224), Image.LANCZOS)
	img224.save("/tmp/gg.png")

	imgtensor = Variable(torch.FloatTensor(np.array(img224).astype(np.float) * 2.0 / 255.0 - 1.0).permute(2, 0, 1).unsqueeze(0), requires_grad=False)
	results = F.softmax(mlp(imgtensor), dim=1)
	probabilities, indices = results.squeeze().sort(descending=True)
	probabilities = probabilities.detach().numpy().tolist()
	indices = indices.detach().numpy().tolist()
	zipped = [(labels[x[0]], x[0], x[1]) for x in zip(indices, probabilities)]
	with open("/tmp/gg.json", "w") as jsonfp:
		json.dump(zipped, jsonfp, indent=4, sort_keys=True)
	for i in xrange(0, 10):
		print zipped[i]
	img.close()

def main(argv):
	resnet = resnet152(pretrained=True)
	resnet.train(False)
	labels = []
	with open("/usr/local/caffe/data/ilsvrc12/synset_words.txt", "r") as wordsfp:
		for line in wordsfp:
			labels.append(line.strip())
	for filename in argv:
		classify(resnet, labels, filename)

	torch.save(resnet, "/tmp/myresnet.pth")
	print resnet

if __name__ == "__main__":
	main(sys.argv[1:])
