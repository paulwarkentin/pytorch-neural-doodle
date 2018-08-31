##
## pytorch-neural-doodle/src/loss/style.py
##
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.

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
		style_normalizer{}:   Norms of style patches for each layer.
	"""

	def __init__(self, style_response, layers, size=(3,3), stride=3):
		"""Initialize a new loss function.

		Arguments:
			style_response (tensor):  Model response to style image.
			layers (list):            Layers for which response is compared.
			size (tuple):             Size of single patch.
			stride (int):             Stride of sliding blocks in spatial dimensions.
		"""
		super().__init__()

		# setup unfold convolution and layers
		self.unfold = nn.Unfold(size, stride=stride)
		self.layers = layers

		for layer in layers:
			# unfold style response
			style_resp = style_response["conv{}".format(layer)]
			style_patches = self.unfold(style_resp)
			(_, patch_count, patch_size) = style_patches.size()
			style_patches = style_patches.view(patch_count, patch_size)
			style_normalizer = torch.reciprocal(torch.norm(style_patches, p=2, dim=1))
			setattr(self, "style_patches{}".format(layer), style_patches)
			setattr(self, "style_normalizer{}".format(layer), style_normalizer)

	
	def loss(self, model_response):
		"""Compute loss of given image w.r.t. initialized style.

		Arguments:
			model_response (tensor): Model response to image.
		"""
		for l in self.layers:
			# collect response patches for this layer
			style_patches = getattr(self, "style_patches{}".format(l))
			(patch_count, patch_size) = style_patches.size()
			img_patches = self.unfold(model_response["conv{}".format(l)]).view(patch_count, patch_size)

			# compute nearest neighbours
			similarity_matrix = img_patches @ style_patches.t()
			img_normalizer = torch.reciprocal(torch.norm(img_patches, p=2, dim=1))
			style_normalizer = getattr(self, "style_normalizer{}".format(l))
			similarity_matrix = img_normalizer.expand(patch_count, patch_count) * similarity_matrix 
			similarity_matrix = similarity_matrix.t() * style_normalizer.expand(patch_count, patch_count)
			nearest_neighbours = torch.argmax(similarity_matrix, dim=0)
			
			# free up some memory
			del similarity_matrix
			del img_normalizer

			print(nearest_neighbours.size())