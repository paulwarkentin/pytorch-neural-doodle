##
## pytorch-neural-doodle/src/utils/common/terminal.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
##

import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.logging import logging_wait


def query_yes_no(question, default = "yes"):
	"""Ask a yes / no question in the console.

	Arguments:
		question: question that is presented to the user.
		default: presumed answer if the user just hits <Enter>. Possible values are 'yes', 'no' and None.

	Returns:
		Flag whether the user answered with 'yes' or 'no'.
	"""
	valid = {
		"yes": True,
		"y": True,
		"ye": True,
		"no": False,
		"n": False
	}

	if default is None:
		prompt = " [yes/no] "
	elif default == "yes":
		prompt = " [Yes/no] "
	elif default == "no":
		prompt = " [yes/No] "
	else:
		raise ValueError("invalid default answer: '%s'" % default)

	# wait for the answer of the user
	while True:
		logging_wait(question + prompt, end="")
		choice = input().lower()
		if default is not None and choice == '':
			return valid[default]
		elif choice in valid:
			return valid[choice]
		else:
			logging_wait("Please respond with 'yes' or 'no' (or 'y' or 'n').")
