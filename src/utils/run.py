##
## pytorch-neural-doodle/src/src/utils/run.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 05/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 16/08/2018.
##

import json
import os
import sys
import uuid
from datetime import datetime

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from utils.common.files import get_full_path, mkdir


class Run(object):
	"""Handle a training run and its configuration.

	Arguments:
		id: Identifier of the run.
		base_path: Absolute path to the base directory of the run.
		config_file_path: Absolute path to the configuration file of the run.
		checkpoints_path: Absolute path to the checkpoints directory.
		checkpoints_file_path: Absolute path to the checkpoints file.
		__open: Flag whether the run was opened.
		__config: Dictionary containing the configuration of the run.
	"""

	def __init__(self, run_id = None):
		"""Initialize the class.

		Arguments:
			run_id: Identifier of the run. Defaults to None.
		"""
		self.id = run_id

		self.base_path = None
		self.config_file_path = None
		self.checkpoints_path = None
		self.checkpoints_file_path = None

		self.__open = False
		self.__config = None


	def open(self):
		"""Open the run and initialize all paths and configurations.

		Returns:
			Flag whether the run was successfully opened.
		"""
		# create a new run id
		if self.id is None:
			while True:
				id = "run_{}_{}".format(
					datetime.now().strftime("%Y-%m-%d-%H-%M"), uuid.uuid4()
				)
				path = get_full_path("runs", id)
				if not os.path.isdir(path):
					break
			self.id = id
			self.base_path = path

		# check whether the run with the initialized id exists
		else:
			base_path = get_full_path("runs", self.id)
			if not os.path.isdir(base_path):
				base_path = get_full_path(self.id)
				if not os.path.isdir(base_path):
					self.__open = False
					return False
				base_path = base_path.rstrip("/")
				self.id = os.path.basename(base_path)
				base_path = get_full_path("runs", self.id)
			self.id = self.id.rstrip("/")
			self.base_path = base_path

		# create paths
		self.config_file_path = os.path.join(self.base_path, "config.json")
		self.checkpoints_path = os.path.join(self.base_path, "checkpoints")
		self.checkpoints_file_path = os.path.join(self.checkpoints_path, "checkpoints")

		mkdir(self.base_path)
		mkdir(self.checkpoints_path)

		# load or initialize configuration
		if os.path.isfile(self.config_file_path):
			with open(self.config_file_path, "r") as file:
				self.__config = json.load(file)
		else:
			self.__config = {}

		self.__open = True
		return True


	def get_config_value(self, *keys):
		"""Get a value from the configuration.

		Arguments:
			keys: Path to the value to retrieve.

		Returns:
			Value at the given path if it exists, otherwise None.
		"""
		if not self.__open or self.__config is None:
			raise RuntimeError("Before accessing a run configuration, the run must be opened.")

		if len(keys) < 1:
			raise RuntimeError("Please provide at least one key.")

		parent = self.__config
		for key in keys:
			if key not in parent:
				return None
			parent = parent[key]

		return parent


	def set_config_value(self, value, *keys):
		"""Update a value in the configuration.

		Arguments:
			value: New value.
			keys: Path to the value to update.
		"""
		if not self.__open or self.__config is None:
			raise RuntimeError("Before modifying a run configuration, the run must be opened.")

		if len(keys) < 1:
			raise RuntimeError("Please provide at least one key.")

		parent = self.__config
		for ii, key in enumerate(keys):
			if ii == len(keys) - 1:
				parent[key] = value
			else:
				parent = parent.setdefault(key, {})


	def get_config(self):
		"""Get the complete configuration.

		Returns:
			A copy of the configuration dictionary.
		"""
		if not self.__open or self.__config is None:
			raise RuntimeError("Before accessing a run configuration, the run must be opened.")

		return self.__config.copy()


	def save_config(self):
		"""Save the configuration to file.
		"""
		if not self.__open:
			raise RuntimeError("Before writing to a run configuration file, the run must be opened.")

		with open(self.config_file_path, "w") as file:
			json.dump(self.__config, file, indent=4)
