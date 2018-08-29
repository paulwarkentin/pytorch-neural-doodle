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

import torch.nn as nn

class StyleLoss(nn.Module):
	"""A PyTorch module that implements loss w.r.t. style for style transfer.

	For information on this loss function, refer to the documentation 
	in `pytorch-neural-doodle/docs` or to Champandard
	"Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork"
	(https://arxiv.org/pdf/1603.01768.pdf).

	Attributes:
		unfold:        Unfolding convolution to compute patches.
		style_patches: Patches of the style image.
	"""

	def __init__(self, style, size=(3,3), dilation=1):
		"""Initialize a new loss function.

		Arguments:
			style (tensor): Style image tensor.
			size (tuple):   Size of single patch.
			dilation (int): Dilation between adjacent patches.
		"""
		super().__init__()

		# setup unfold convolution
		self.unfold = torch.nn.Unfold(size, dilation=dilation)

		# unfold style image
		self.style_patches = self.unfold(style)
		