##
## pytorch-neural-doodle/src/utils/common/logging.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
##

import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)


prefix_name = "torchdoodle"


def logging(*args, **kwargs):
	"""Prints a message to the console and flashes the output.

	Arguments:
		args: List of arguments passed to the print function.
		kwargs: Dictionary of arguments passed to the print function.
	"""
	print(*args, **kwargs)
	sys.stdout.flush()


def logging_info(*args, **kwargs):
	"""Prints an information message to the console with an app prefix.

	Arguments:
		args: List of arguments passed to the print function.
		kwargs: Dictionary of arguments passed to the print function.
	"""
	__logging("INFO", *args, **kwargs)


def logging_wait(*args, **kwargs):
	"""Prints a waiting message to the console with an app prefix.

	Arguments:
		args: List of arguments passed to the print function.
		kwargs: Dictionary of arguments passed to the print function.
	"""
	__logging("WAIT", *args, **kwargs)


def logging_warn(*args, should_exit = False, **kwargs):
	"""Prints a warning message to the console with an app prefix.

	Arguments:
		args: List of arguments passed to the print function.
		should_exit: Flag whether to exit the application after printing the message. Defaults to False.
		kwargs: Dictionary of arguments passed to the print function.
	"""
	__logging("WARNING", *args, **kwargs)
	if should_exit:
		exit()


def logging_error(*args, should_exit = True, **kwargs):
	"""Prints an error message to the console with an app prefix.

	Arguments:
		args: List of arguments passed to the print function.
		should_exit: Flag whether to exit the application after printing the message. Defaults to True.
		kwargs: Dictionary of arguments passed to the print function.
	"""
	__logging("ERROR", *args, **kwargs)
	if should_exit:
		exit()


def __logging(preprefix, *args, with_prefix = True, **kwargs):
	"""Prints a message to the console with a prefix and flashes the output.

	Arguments:
		preprefix: Pre-prefix of the message.
		args: List of arguments passed to the print function.
		with_prefix: Flag whether to print the message with a prefix. Defaults to True.
		kwargs: Dictionary of arguments passed to the print function.
	"""
	print("{}:".format(preprefix), end="")
	if with_prefix:
		print("{}:".format(prefix_name), end="")
	print(*args, **kwargs)
	sys.stdout.flush()
