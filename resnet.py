#! /usr/bin/python

"""This is basically my implementation of ResNet. Why?! There's already torchvision.ResNet!!
  Because
	A. This will give me a better understanding of how it works,
	B. I can futz with it's internals (maybe swapping ReLU's for RReLU's, adding dropout for instance) and
	C. I can try to build it with an eye towards TensorFlow interoperabilitiy.

   It'll _probably_ be mostly _copied_ from torchvision.ResNet tho.
   (Also, my anal nature made it be pylint clean.)
"""

import math
import torch
import torch.nn as nn
from torch.utils import model_zoo

MODEL_URLS = { \
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', \
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', \
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', \
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', \
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', \
}

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


#pylint: disable=too-many-instance-attributes
class BasicBlock(nn.Module):
	expansion = 1

	#pylint: disable=too-many-arguments
	def __init__(self, inplanes, planes, stride=1, downsample=None, activation=nn.ReLU, dropout_rate=0.5):
		super(BasicBlock, self).__init__()
		self.dropout2d = nn.Dropout2d(dropout_rate)
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.activation = activation()
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
	#pylint: enable=too-many-arguments

	def forward(self, *args):
		inp = args[0]
		residual = inp

		out = self.conv1(inp)
		out = self.bn1(out)
		out = self.activation(out)
		out = self.dropout2d(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(inp)

		out += residual
		out = self.activation(out)
		out = self.dropout2d(out)

		return out
#pylint: enable=too-many-instance-attributes

#pylint: disable=too-many-instance-attributes
class Bottleneck(nn.Module):
	expansion = 4

	#pylint: disable=too-many-arguments
	def __init__(self, inplanes, planes, stride=1, downsample=None, activation=nn.ReLU, dropout_rate=0.5):
		super(Bottleneck, self).__init__()
		self.dropout2d = nn.Dropout2d(dropout_rate)
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.activation = activation()
		self.downsample = downsample
		self.stride = stride
	#pylint: enable=too-many-arguments

	def forward(self, *args):
		inp = args[0]
		residual = inp

		out = self.conv1(inp)
		out = self.bn1(out)
		out = self.activation(out)
		out = self.dropout2d(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.activation(out)
		out = self.dropout2d(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(inp)

		out += residual
		out = self.activation(out)
		out = self.dropout2d(out)

		return out
#pylint: enable=too-many-instance-attributes

#pylint: disable=too-many-instance-attributes
class ResNet(nn.Module):
	#pylint: disable=too-many-arguments
	def __init__(self, block, layers, num_classes=1000, activation=nn.ReLU, dropout_rate=0.5):
		self.inplanes = 64
		self.block = block
		super(ResNet, self).__init__()
		self.dropout2d = nn.Dropout2d(dropout_rate)
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.activation = activation()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(64, layers[0])
		self.layer2 = self._make_layer(128, layers[1], stride=2)
		self.layer3 = self._make_layer(256, layers[2], stride=2)
		self.layer4 = self._make_layer(512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.dense = nn.Linear(512 * self.block.expansion, num_classes)
		self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
		self.std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				size = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
				module.weight.data.normal_(0, math.sqrt(2. / size))
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()
	#pylint: enable=too-many-arguments

	def _make_layer(self, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * self.block.expansion:
			downsample = nn.Sequential( \
				nn.Conv2d(self.inplanes, planes * self.block.expansion, kernel_size=1, stride=stride, bias=False), \
				nn.BatchNorm2d(planes * self.block.expansion), \
			)

		layers = []
		layers.append(self.block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * self.block.expansion
		for _ in range(1, blocks):
			layers.append(self.block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, *args):
		inp = args[0]

# from https://pytorch.org/docs/stable/torchvision/models.html
# All pre-trained models expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where
# H and W are expected to be at least 224. The images have to be loaded
# in to a range of [0, 1] and then normalized using
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# You can use the following transform to normalize:
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Binesh - Fine. But, *I* want the inputs to be between -1 and 1 with 0 mean.
# So. Let's add a translation here. Really, I guess it's just (inp+1) / 2 to make it 0..1
# Then, we just use the normalize as they specify.
# Although, if this doesn't perform gangbusters out the box, I'm deleting this line
# right quick.

# Ok, actually, that's pretty damned good. Fine, I hate it, but I'm keeping it.

		inp = (((inp + 1.0) / 2.0) - self.mean.type_as(inp)) / self.std.type_as(inp)

		out = self.conv1(inp)
		out = self.bn1(out)
		out = self.activation(out)
		out = self.dropout2d(out)
		out = self.maxpool(out)

		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)

		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = self.dense(out)

		return out
#pylint: enable=too-many-instance-attributes

def zoo_filter(url):
	data = model_zoo.load_url(url)
	assert "fc.weight" in data
	assert "fc.bias" in data
	data["dense.weight"] = data["fc.weight"]
	data["dense.bias"] = data["fc.bias"]
	del data["fc.weight"]
	del data["fc.bias"]
	return data

def resnet18(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained:
		model.load_state_dict(zoo_filter(MODEL_URLS['resnet18']))
	return model



def resnet34(pretrained=False, **kwargs):
	"""Constructs a ResNet-34 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(zoo_filter(MODEL_URLS['resnet34']))
	return model



def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(zoo_filter(MODEL_URLS['resnet50']))
	return model



def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(zoo_filter(MODEL_URLS['resnet101']))
	return model



def resnet152(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(zoo_filter(MODEL_URLS['resnet152']))
	return model
