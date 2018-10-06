##
## pytorch-neural-doodle/src/utils/image.py
##
## Created by Bastian Boll <mail@bbboll.com> on 29/08/2018.
## Updated by Bastian Boll <mail@bbboll.com> on 06/10/2018.

import os.path
import numpy as np
import sys
import torch
from skimage import io

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.logging import logging_error


def load(filepath, device=torch.device("cpu")):
	"""Load image from filepath as tensor.

	Arguments:
		filepath (string):      Path to image.
		device (torch device):  Device on which to allocate the image tensor.

	Returns:
		Tensor of image content with shape (1, channels, height, width).
	"""
	if not os.path.isfile(filepath):
		logging_error("File {} does not exist.".format(filepath), should_exit = True)

	image = io.imread(filepath).transpose((2, 0, 1)) / (256.0 / 2.0) - 1.0
	return torch.from_numpy(image).double().unsqueeze(0).to(device)


def save(filepath, image):
	"""Save a given tensor to a filepath.

	Arguments:
		filepath (string): Path to save to.
		image (tensor):    Image to be saved.
	"""
	image = image.detach().cpu().numpy()
	_, channels, height, width = image.shape
	image = image.reshape((channels, height, width)).transpose((1, 2, 0))
	image = (image + 1.0) * (256.0 / 2.0)
	io.imsave(filepath, image.astype("uint8"))
