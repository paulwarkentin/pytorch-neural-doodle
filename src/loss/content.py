##
## pytorch-neural-doodle/src/loss/content.py
##
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.

import os.path
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

import torch
import torch.nn as nn

class ContentLoss(nn.Module):
	"""A PyTorch module that implements loss w.r.t. content for style transfer.

	For information on this loss function, refer to the documentation 
	in `pytorch-neural-doodle/docs` or to Gatys et al
	"A neural algorithm of artistic style"
	(https://arxiv.org/pdf/1508.06576.pdf).

	Attributes:
		-
	"""

	def __init__(self, content):
		"""
		"""
		super().__init__()


		