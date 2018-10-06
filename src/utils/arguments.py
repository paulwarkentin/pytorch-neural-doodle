##
## pytorch-neural-doodle/src/utils/arguments.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 06/10/2018.
##

import argparse
import os
import sys


def parse_arguments():
	"""Parse CLI arguments.
	"""
	# parse arguments
	parser = argparse.ArgumentParser(
		description = "Semantic Style Transfer after https://arxiv.org/abs/1603.01768."
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
		"--output-map-file",
		default = None,
		type = str,
		required = True,
		help = "Path to the output map file."
	)
	parser.add_argument(
		"--output-content-file",
		default = None,
		type = str,
		required = False,
		help = "Path to the content file."
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
		"--map-channel-weight",
		default = 50.0,
		type = positive_float_type,
		required = False,
		help = "Weight for map channels (gamma)."
	)
	parser.add_argument(
		"--num-phases",
		default = 150,
		type = exclusive_positive_int_type,
		required = False,
		help = "The number of phases to process."
	)
	parser.add_argument(
		"--device",
		default = None,
		type = str,
		required = False,
		help = "Device to train on."
	)
	parser.add_argument(
		"--save-interval",
		default = None,
		type = positive_int_type_or_none,
		required = False,
		help = "Iteration interval for saving the output."
	)
	parser.add_argument(
		"--plot",
		default = False,
		type = bool,
		required = False,
		help = "Boolean flag whether to show live plots."
	)

	arguments = parser.parse_args()
	return arguments


def boolean_type(arg):
	"""Check an argument for a boolean type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	if arg.lower() in ("yes", "true", "t", "y", "1"):
		return True
	if arg.lower() in ("no", "false", "f", "n", "0"):
		return False
	raise argparse.ArgumentTypeError("invalid boolean value: '{}'".format(arg))


def exclusive_positive_int_type(arg):
	"""Check an argument for an exclusive positive integer type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_int = int(arg)
	if arg_int <= 0:
		raise argparse.ArgumentTypeError("invalid positive_int_type value: '{}', must be exclusive positive".format(arg))
	return arg_int


def positive_int_type(arg):
	"""Check an argument for a positive integer type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_int = int(arg)
	if arg_int < 0:
		raise argparse.ArgumentTypeError("invalid positive_int_type value: '{}', must be positive".format(arg))
	return arg_int

def positive_int_type_or_none(arg):
	"""Check an argument for a positive integer type or None.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	if arg == 'None':
		return None
	arg_int = int(arg)
	if arg_int < 0:
		raise argparse.ArgumentTypeError("invalid positive_int_type value: '{}', must be positive".format(arg))
	return arg_int


def exclusive_positive_float_type(arg):
	"""Check an argument for an exclusive positive float type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_float = float(arg)
	if arg_float <= 0.0:
		raise argparse.ArgumentTypeError("invalid positive_float_type value: '{}', must be exclusive positive".format(arg))
	return arg_float


def positive_float_type(arg):
	"""Check an argument for a positive float type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_float = float(arg)
	if arg_float < 0.0:
		raise argparse.ArgumentTypeError("invalid positive_float_type value: '{}', must be positive".format(arg))
	return arg_float


def exclusive_unit_float_type(arg):
	"""Check an argument for an exclusive unit float type.

	Arguments:
		arg: Argument value.

	Returns:
		Argument of valid type.
	"""
	arg_float = float(arg)
	if arg_float <= 0.0 or arg_float >= 1.0:
		raise argparse.ArgumentTypeError("invalid exclusive_unit_float_type value: '{}', must be exclusive between 0 and 1".format(arg))
	return arg_float
