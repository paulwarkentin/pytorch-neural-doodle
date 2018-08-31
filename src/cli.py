##
## pytorch-neural-doodle/src/cli.py
##
## Created by Bastian Boll <mail@bbboll.com> on 31/08/2018.
## Updated by Bastian Boll <mail@bbboll.com> on 31/08/2018.
##

import argparse
from utils.arguments import positive_float_type, exclusive_positive_int_type

def parse_arguments():
	"""Parse CLI arguments.
	"""
	# parse arguments
	parser = argparse.ArgumentParser(
		description = "Generate a new image."
	)
	parser.add_argument(
		"--input-style-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the input style file."
	)
	parser.add_argument(
		"--input-map-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the input map file."
	)
	parser.add_argument(
		"--output-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the output file to be generated."
	)
	parser.add_argument(
		"--output-content-file",
		default = None,
		type = str,
		required = False,
		help = "Path to the content file."
	)
	parser.add_argument(
		"--output-map-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the output map file."
	)
	parser.add_argument(
		"--content-weight",
		default = 10.0,
		type = positive_float_type,
		required = False,
		help = "Weight of the content relative to the style (alpha)."
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
		"--style-layers",
		nargs = "+",
		default = ["3_1", "4_1"],
		type = positive_float_type,
		required = False,
		help = "The layer to use for the style."
	)
	parser.add_argument(
		"--style-weight",
		default = 25.0,
		type = positive_float_type,
		required = False,
		help = "Weight of the style relative to the style (beta)."
	)
	parser.add_argument(
		"--num-phases",
		default = 3,
		type = exclusive_positive_int_type,
		required = False,
		help = "The number of phases to process."
	)
	parser.add_argument(
		"--map-channel-weight",
		default = 50.0,
		type = positive_float_type,
		required = False,
		help = "Weight for map channels (gamma)."
	)
	arguments = parser.parse_args()
	return arguments