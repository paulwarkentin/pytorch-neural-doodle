##
## pytorch-neural-doodle/src/utils/image.py
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
from skimage import io
from .common.logging import logging_error 

def load(filepath):
	"""Load image from filepath as tensor.

	Arguments:
		filepath: Path to image.

	Returns:
		Tensor of image content with shape (1, channels, height, width).
	"""
	if not os.path.isfile(filepath):
		logging_error("File {} does not exist.".format(filepath), should_exit = True)
	image = io.imread(filepath).transpose((2,0,1))
	return torch.from_numpy(image).double().unsqueeze(0)