##
## pytorch-neural-doodle/src/loss/style.py
##
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.
## Created by Bastian Boll <mail@bbboll.com> on 01/09/2018.

import os.path
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

import numpy as np
import torch
import torch.nn as nn

class StyleLoss(nn.Module):
	"""A PyTorch module that implements loss w.r.t. style for style transfer.

	For information on this loss function, refer to the documentation 
	in `pytorch-neural-doodle/docs` or to Champandard
	"Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork"
	(https://arxiv.org/pdf/1603.01768.pdf).

	Attributes:
		unfold:               Unfolding convolution to compute patches.
		layers:               Layers for which response is compared.
		style_patches{}:      Patches of the style image for each layer.
		output_maps{}:        Downsampled output maps for each layer.
		style_normalizer{}:   Norms of style patches for each layer.
	"""

	def __init__(self, style_response, style_map, output_map, layers, map_channel_weight, size=(3,3), stride=3):
		"""Initialize a new loss function.

		Arguments:
			style_response (tensor):  Model response to style image.
			style_map (tensor):       Map for input style.
			output_map (tensor):      Map for output image.
			layers (list):            Layers for which response is compared.
			size (tuple):             Size of single patch.
			stride (int):             Stride of sliding blocks in spatial dimensions.
		"""
		super().__init__()

		# setup unfold convolution and layers
		self.unfold = nn.Unfold(size, stride=stride)
		self.layers = layers

		for layer in layers:
			style_resp = style_response["conv{}".format(layer)]
			patch_height = style_resp.size()[2]
			channel_count = style_resp.size()[0]

			# downsample style maps
			down = int(style_map.size()[2] / patch_height)
			style_map_down = nn.AvgPool2d((down, down))(style_map)
			output_map_down = nn.AvgPool2d((down, down))(output_map)

			# unfold style response
			style_resp = torch.cat((style_resp, map_channel_weight*style_map_down*channel_count), dim=1)
			style_patches = torch.squeeze(self.unfold(style_resp))

			# compute normalizer for cosine distance
			style_normalizer = torch.reciprocal(torch.norm(style_patches, p=2, dim=0))

			setattr(self, "style_patches{}".format(layer), style_patches)
			setattr(self, "output_maps{}".format(layer), map_channel_weight*output_map_down*channel_count)
			setattr(self, "style_normalizer{}".format(layer), style_normalizer)

	
	def loss(self, model_response):
		"""Compute loss of given image w.r.t. initialized style.

		Arguments:
			model_response (tensor): Model response to target image.
		"""
		losses = []
		for l in self.layers:
			# collect response patches for this layer
			style_patches = getattr(self, "style_patches{}".format(l))
			(patch_size, patch_count) = style_patches.size()
			output_map = getattr(self, "output_maps{}".format(l))
			img_patches = model_response["conv{}".format(l)]
			img_patches = torch.cat((img_patches, output_map), dim=1)
			img_patches = torch.squeeze(self.unfold(img_patches))

			# compute nearest neighbours
			similarity_matrix = img_patches.t() @ style_patches
			img_normalizer = torch.reciprocal(torch.norm(img_patches, p=2, dim=0))
			style_normalizer = getattr(self, "style_normalizer{}".format(l))
			similarity_matrix = img_normalizer.expand(patch_count, patch_count) * similarity_matrix 
			similarity_matrix = similarity_matrix.t() * style_normalizer.expand(patch_count, patch_count)
			nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
			
			# free up some memory
			del similarity_matrix
			del img_normalizer

			losses.append(nn.functional.mse_loss(img_patches, style_patches[:,nearest_neighbours]))
		return sum(losses)