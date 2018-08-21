##
## pytorch-neural-doodle/src/extract_vgg_19_weights.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 06/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 06/08/2018.
##

import argparse
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.files import get_full_path
from utils.common.logging import logging_info
from utils.common.terminal import query_yes_no


if __name__ == "__main__":

	# parse arguments
	parser = argparse.ArgumentParser(
		description = "Export the pre-trained weights of the VGG 19 network."
	)
	parser.add_argument(
		"--checkpoint-file",
		default = get_full_path("models", "vgg_19_imagenet", "vgg_19.ckpt"),
		type = str,
		required = False,
		help = "Path to the checkpoint file to extract weights from."
	)
	arguments = parser.parse_args()

	# print some information
	logging_info("Export the pre-trained weights of the VGG 19 network.")
	logging_info("Checkpoint file: {}".format(arguments.checkpoint_file))

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	logging_info("Read weights original checkpoint file.")

	# initialize checkpoint reader
	reader = tf.train.NewCheckpointReader(arguments.checkpoint_file)

	# read weights
	data = {}
	for block_ii, num_convs in enumerate([2, 2, 4, 4, 4]):
		block_name = "conv{}".format(block_ii + 1)
		for conv_ii in range(num_convs):
			conv_name = "{}_{}".format(block_name, conv_ii + 1)
			full_name = "vgg_19/{}/{}".format(block_name, conv_name)
			data[conv_name] = [
				reader.get_tensor("{}/weights".format(full_name)),
				reader.get_tensor("{}/biases".format(full_name))
			]

	# transpose weights for PyTorch
	for name in data:
		data[name][0] = np.transpose(data[name][0], (3, 2, 0, 1))

	logging_info("Save exported weights to a new file.")

	# save weights
	name, _ = os.path.splitext(arguments.checkpoint_file)
	output_filename = "{}.minimum.pkl".format(name)
	with open(output_filename, "wb") as file:
		pickle.dump(data, file)

	logging_info("All weights are successfully saved to '{}'.".format(output_filename))
