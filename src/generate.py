##
## pytorch-neural-doodle/src/generate.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 10/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 20/08/2018.
##

import argparse
import numpy as np
import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.arguments import positive_float_type, exclusive_positive_int_type
from utils.common.files import get_full_path
from utils.common.logging import logging_info
from utils.common.terminal import query_yes_no


if __name__ == "__main__":

	# parse arguments
	parser = argparse.ArgumentParser(
		description = "Generate a new image."
	)
	parser.add_argument(
		"--input-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the input file."
	)
	parser.add_argument(
		"--input-style-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the input style file."
	)
	parser.add_argument(
		"--output-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the output file."
	)
	parser.add_argument(
		"--output-style-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the output style file."
	)
	parser.add_argument(
		"--content-weight",
		default = 10.0,
		type = positive_float_type,
		required = False,
		help = "Weight of the content relative to the style."
	)
	parser.add_argument(
		"--content-layers",
		nargs = "+",
		default = ["4_2"],
		type = positive_float_type,
		required = False,
		help = "The layer to use for the content."
	)
	parser.add_argument(
		"--style-weight",
		default = 25.0,
		type = positive_float_type,
		required = False,
		help = "Weight of the style relative to the style."
	)
	parser.add_argument(
		"--style-layers",
		nargs = "+",
		default = ["3_1", "4_1"],
		type = positive_float_type,
		required = False,
		help = "The layer to use for the style."
	)
	parser.add_argument(
		"--num-phases",
		default = 3,
		type = exclusive_positive_int_type,
		required = False,
		help = "The number of phases to process."
	)
	arguments = parser.parse_args()

	# print some information
	logging_info("Generate a new image.")
	logging_info("Input file:        {}".format(arguments.input_file))
	logging_info("Input style file:  {}".format(arguments.input_style_file))
	logging_info("Output file:       {}".format(arguments.output_file))
	logging_info("Output style file: {}".format(arguments.output_style_file))
	logging_info("Content weight:    {}".format(arguments.content_weight))
	logging_info("Content layers:    {}".format(arguments.content_layers))
	logging_info("Style weight:      {}".format(arguments.style_weight))
	logging_info("Style layers:      {}".format(arguments.style_layers))

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	# initialize model
	model = GenerativeModel(input_channels=3)

	vgg_19_path = get_full_path("models", "vgg_19_imagenet", "vgg_19.minimum.pkl")
	if not os.path.isfile(vgg_19_path):
		logging_error("Please download the weights and biases of the VGG 19 network from " \
					  "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz, extract the archive and run the Python script "\
					  "`extract_vgg_19_weights.py`.")
	model.initialize(vgg_19_path)

	# main loop
	for ii in range(arguments.num_phases):
		pass
