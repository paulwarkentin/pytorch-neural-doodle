##
## pytorch-neural-doodle/src/models/vgg16.py
##
## Created by Bastian Boll <mail@bbboll.com> on 05/09/2018.
## Updated by Bastian Boll <mail@bbboll.com> on 05/09/2018.
##

import pickle
import torch
import torch.nn as nn

import sys
import os.path

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


class VGG16(nn.Module):
	"""A PyTorch module that implements the VGG 16 network for image classification.

	For information on the VGG 16 network see Simonyan & Zisserman
	"Very Deep Convolutional Networks for Large-Scale Image Recognition"
	(https://arxiv.org/abs/1409.1556).

	Attributes:
		conv{x}_{y}: Convolutional modules of the VGG 16 network.
		pool:        Pooling module.
		relu:        ReLU activation module.
		map{x}:      Pooling modules for the style image.
		nn{x}_{y}:   Convolutional modules for nearest neighbor calculation.
	"""

	def __init__(self, input_channels = 3):
		"""Initialize a new generative model.

		Arguments:
			input_channels (int): Number of input channels.
		"""
		super().__init__()

		self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=1)
		self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1)

		self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

		self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

		self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

		self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

		self.pool = nn.MaxPool2d(2, stride=2)
		self.relu = nn.ReLU()

		self.map1 = nn.MaxPool2d(1)
		self.map2 = nn.MaxPool2d(2)
		self.map3 = nn.MaxPool2d(4)
		self.map4 = nn.MaxPool2d(8)
		self.map5 = nn.MaxPool2d(16)

		self.nn1_1 = nn.Conv2d( 64, 1, 3)
		self.nn1_2 = nn.Conv2d( 64, 1, 3)

		self.nn2_1 = nn.Conv2d(128, 1, 3)
		self.nn2_2 = nn.Conv2d(128, 1, 3)

		self.nn3_1 = nn.Conv2d(256, 1, 3)
		self.nn3_2 = nn.Conv2d(256, 1, 3)
		self.nn3_3 = nn.Conv2d(256, 1, 3)

		self.nn4_1 = nn.Conv2d(512, 1, 3)
		self.nn4_2 = nn.Conv2d(512, 1, 3)
		self.nn4_3 = nn.Conv2d(512, 1, 3)

		self.nn5_1 = nn.Conv2d(512, 1, 3)
		self.nn5_2 = nn.Conv2d(512, 1, 3)
		self.nn5_3 = nn.Conv2d(512, 1, 3)


	def initialize(self, filename):
		"""Load and initialize weights and biases of the VGG 16 network.

		Arguments:
			filename (str): Filename of the file containing the weights and biases.
		"""
		# load weights and biases
		with open(filename, "rb") as file:
			data = pickle.load(file)

		# initialize parameters
		for name in data:
			module = getattr(self, name)
			module.weight.data = torch.from_numpy(data[name][0]).double()
			module.bias.data = torch.from_numpy(data[name][1]).double()


	def forward(self, image_input, extract_layers=["out"]):
		"""Do forward pass of the network.

		Arguments:
			image_input: Original input image.

		Returns:
			dict: Output tensors.
		"""
		outputs = {}

		x = image_input
		for block_ii, num_convs in enumerate([2, 2, 3, 3, 3]):
			conv_block_name = "conv{}".format(block_ii + 1)
			nn_block_name = "nn{}".format(block_ii + 1)

			for conv_ii in range(num_convs):
				conv_name = "{}_{}".format(conv_block_name, conv_ii + 1)
				conv_ind = "{}_{}".format(block_ii + 1, conv_ii + 1)
				x = getattr(self, conv_name)(x)
				x = self.relu(x)
				if conv_ind in extract_layers:
					outputs[conv_name] = x

			x = self.pool(x)
		
		if "out" in extract_layers:
			outputs["out"] = x
		
		return outputs
