##
## pytorch-neural-doodle/src/utils/common/files.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
##

import errno
import os
import sys

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.static import static_vars


@static_vars(root_path=None)
def root_dir():
	"""Make absolute path to root directory of the project.

	Returns:
		Absolute path of project.
	"""
	if root_dir.root_path is None:
		path = os.path.abspath(__file__) # e.g. `.../xyc/src/utils/common/files.py`
		path = os.path.dirname(path) # e.g. `.../xyc/src/utils/common/`
		path = os.path.dirname(path) # e.g. `.../xyc/src/utils/`
		path = os.path.dirname(path) # e.g. `.../xyc/src/`
		root_dir.root_path = os.path.dirname(path) # e.g. `.../xyc/`
	return root_dir.root_path


def get_full_path(*path):
	"""Make absolute path to a file or directory in the project folder.

	Arguments:
		path: List of path elements.

	Returns:
		Absolute path to requested file or directory.
	"""
	return os.path.join(root_dir(), *path)


def get_real_path(*path):
	"""Make absolute path to a file or directory in the project folder. All symbolic links are resolved.

	Arguments:
		path: List of path elements.

	Returns:
		Absolute path to requested file or directory.
	"""
	return os.path.realpath(os.path.join(root_dir(), *path))


def mkdir(*path, mode = 0o777, recursive = True, ignore_existing = True):
	"""Make recursive directories to a given path.

	Arguments:
		path: List of path elements.
		mode: Mode of the new directory. Defaults to 0777.
		recursive: Flag whether to create all non-existing parent directories. Defaults to True.
		ignore_existing: Flag whether to ignore existing destination directory. Defaults to True.
	"""
	path = os.path.join(*path)
	try:
		if recursive:
			os.makedirs(path, mode=mode)
		else:
			os.mkdir(path, mode=mode)
	except OSError as error:
		if not ignore_existing or error.errno != errno.EEXIST or not os.path.isdir(path):
			raise


def get_files(*path):
	"""Get files in a given directory.

	Arguments:
		path: List of path elements to destination directory.

	Returns:
		List of files in the destination directory.
	"""
	path = os.path.join(*path)
	items = os.listdir(path)
	files = []
	for item in items:
		item_path = os.path.join(path, item)
		if os.path.isfile(item_path):
			files.append(item)
	return files


def get_directories(*path):
	"""Get directories in a given directory.

	Arguments:
		path: List of path elements to destination directory.

	Returns:
		List of directories in the destination directory.
	"""
	path = os.path.join(*path)
	items = os.listdir(path)
	files = []
	for item in items:
		item_path = os.path.join(path, item)
		if os.path.isdir(item_path):
			files.append(item)
	return files
