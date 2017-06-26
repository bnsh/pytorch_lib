#! /usr/bin/python

"""This will do some basic tests of AlexNet"""

import sys
import struct
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pytorchlib.AlexNet import AlexNet

def read_string(packedfp):
	sizebytes = packedfp.read(4)
	size, = struct.unpack("<I", sizebytes)
	return packedfp.read(size)

def read_int(packedfp):
	sizebytes = packedfp.read(4)
	size, = struct.unpack("<I", sizebytes)
	return size

def read_long(packedfp):
	sizebytes = packedfp.read(8)
	size, = struct.unpack("<Q", sizebytes)
	return size

def read_data(packedfp, size):
	sizebytes = packedfp.read(8*size)
	data = struct.unpack("<" + "d" * size, sizebytes)
	return data

def read_mlp(label, filename, mlp):
	with open(filename, "r") as packedfp:
		_ = read_string(packedfp) # the data type, but we're not using that.
		_ = read_int(packedfp) # Dimensions, but we're not using that either.
		size = read_long(packedfp)
		data = torch.Tensor(np.array(read_data(packedfp, size)))
		mlp.weight.data.copy_(data[0:mlp.weight.numel()])
		mlp.bias.data.copy_(data[mlp.weight.numel():])
		print label, (mlp.weight.numel() + mlp.bias.numel()) - len(data)

#pylint: disable=too-many-locals
def classify(mlp, labels, filename):
	img = Image.open(filename)
	width, height = img.size
	size = min(width, height)
	left = (width-size) / 2
	top = (height-size) / 2
	img224 = img.crop((left, top, left+size, top+size)).resize((224, 224), Image.LANCZOS)
	img224.save("/tmp/gg.png")

	imgtensor = Variable(torch.FloatTensor(np.array(img224).astype(np.float) - 127.5).permute(2, 0, 1).unsqueeze(0), requires_grad=False)
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
	alexnet = AlexNet(None)
	alexnet.train(False)
	read_mlp("alexnet.conv1.conv2d", "/tmp/bnet/layer-1-1.bin", alexnet.conv1.conv2d)
	read_mlp("alexnet.conv2.conv1", "/tmp/bnet/layer-2-2-1.bin", alexnet.conv2.conv1)
	read_mlp("alexnet.conv2.conv2", "/tmp/bnet/layer-2-2-2.bin", alexnet.conv2.conv2)
	read_mlp("alexnet.conv3.conv2d", "/tmp/bnet/layer-3-1.bin", alexnet.conv3.conv2d)
	read_mlp("alexnet.conv4.conv1", "/tmp/bnet/layer-4-2-1.bin", alexnet.conv4.conv1)
	read_mlp("alexnet.conv4.conv2", "/tmp/bnet/layer-4-2-2.bin", alexnet.conv4.conv2)
	read_mlp("alexnet.conv5.conv1", "/tmp/bnet/layer-5-2-1.bin", alexnet.conv5.conv1)
	read_mlp("alexnet.conv5.conv2", "/tmp/bnet/layer-5-2-2.bin", alexnet.conv5.conv2)
	read_mlp("alexnet.fc6.linear", "/tmp/bnet/layer-6-2.bin", alexnet.fc6.linear)
	read_mlp("alexnet.fc7.linear", "/tmp/bnet/layer-7-1.bin", alexnet.fc7.linear)
	read_mlp("alexnet.fc8.linear", "/tmp/bnet/layer-8-1.bin", alexnet.fc8.linear)
	labels = []
	with open("/usr/local/caffe/data/ilsvrc12/synset_words.txt", "r") as wordsfp:
		for line in wordsfp:
			labels.append(line.strip())
	for filename in argv:
		classify(alexnet, labels, filename)

if __name__ == "__main__":
	main(sys.argv[1:])
