##
## pytorch-neural-doodle/src/loss/content.py
##
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 06/10/2018.

import os.path
import sys
import torch
import torch.nn as nn


class ContentLoss(nn.Module):
	"""A PyTorch module that implements loss w.r.t. content for style transfer.

	For information on this loss function, refer to the documentation in `pytorch-neural-doodle/docs` or to Gatys et al "A neural
	algorithm of artistic style" (https://arxiv.org/pdf/1508.06576.pdf).
	"""

	def __init__(self, content_response, layers):
		super().__init__()
		self.layers = layers
		for layer in layers:
			content_resp = content_response["conv{}".format(layer)]
			setattr(self, "content_response{}".format(layer), content_resp)

	def loss(self, model_response):
		"""Compute loss of given image w.r.t. initialized content.

		Arguments:
			model_response (tensor): Model response to target image.
		"""
		losses = []
		for l in self.layers:
			content_response = getattr(self, "content_response{}".format(l))
			target_response = model_response["conv{}".format(l)]
			losses.append(nn.functional.mse_loss(content_response, target_response))

		return sum(losses)
