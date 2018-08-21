##
## pytorch-neural-doodle/src/src/utils/arguments.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
## Created by Paul Warkentin <paul@warkentin.email> on 16/08/2018.
##

import argparse
import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.logging import logging_error


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
