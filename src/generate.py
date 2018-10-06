##
## pytorch-neural-doodle/src/generate.py
##
## Created by Paul Warkentin <paul@warkentin.email> on 10/08/2018.
## Updated by Paul Warkentin <paul@warkentin.email> on 06/10/2018.
##

import atexit
import numpy as np
import os
import shutil
import sys
import time
import torch
import torch.nn as nn

__exec_dir = sys.path[0]
while os.path.basename(__exec_dir) != "src":
	__exec_dir = os.path.dirname(__exec_dir)
	sys.path.insert(0, __exec_dir)

from loss import ContentLoss, StyleLoss
from models import VGG19
from utils import LivePlot, Run, image
from utils.arguments import parse_arguments
from utils.common.files import get_full_path
from utils.common.logging import logging_info
from utils.common.terminal import query_yes_no


def exit_handler(run):
	"""Function that is called before the script exits.

	Arguments:
		run: The run that needs exit handling.
	"""
	force_drop_run = True

	# remove run if no image was saved
	output_files = os.listdir(run.output_path)
	if len(output_files) > 0:
		force_drop_run = False

	if force_drop_run and os.path.exists(run.base_path):
		shutil.rmtree(run.base_path)

	# skip if run does not exist
	if not os.path.exists(run.base_path):
		return

	# ask user to keep the run
	should_keep_run = query_yes_no(
		"Should the run '{}' be kept?".format(os.path.basename(run.base_path)),
		default="yes"
	)
	if not should_keep_run:
		shutil.rmtree(run.base_path)


if __name__ == "__main__":
	# get arguments from console
	arguments = parse_arguments()

	# initialize new run
	run = Run(run_id=None)
	run.open()

	run.set_config_value(arguments.input_style_file, "files", "input_style")
	run.set_config_value(arguments.input_map_file, "files", "input_map")
	run.set_config_value(arguments.output_map_file, "files", "output_map")
	run.set_config_value(arguments.output_content_file, "files", "output_content")
	run.set_config_value(arguments.content_weight, "training", "content_weight")
	run.set_config_value(arguments.content_layers, "training", "content_layers")
	run.set_config_value(arguments.style_layers, "training", "style_layers")
	run.set_config_value(arguments.style_weight, "training", "style_weight")
	run.set_config_value(arguments.map_channel_weight, "training", "map_channel_weight")
	run.set_config_value(arguments.num_phases, "training", "num_phases")
	run.set_config_value(arguments.device, "training", "device")
	run.set_config_value(arguments.save_interval, "output", "save_interval")
	run.set_config_value(arguments.plot, "output", "plot")
	run.save_config()

	# print some information
	logging_info("Semantic Style Transfer after https://arxiv.org/abs/1603.01768.")
	logging_info("Input style file:    {}".format(arguments.input_style_file))
	logging_info("Input map file:      {}".format(arguments.input_map_file))
	logging_info("Output map file:     {}".format(arguments.output_map_file))
	logging_info("Output content file: {}".format(arguments.output_content_file))
	logging_info("Content weight:      {}".format(arguments.content_weight))
	logging_info("Content layers:      {}".format(arguments.content_layers))
	logging_info("Style layers:        {}".format(arguments.style_layers))
	logging_info("Style weight:        {}".format(arguments.style_weight))
	logging_info("Map channel weight:  {}".format(arguments.map_channel_weight))
	logging_info("Num phases:          {}".format(arguments.num_phases))
	logging_info("Device:              {}".format(arguments.device))
	logging_info("Save interval:       {}".format(arguments.save_interval or "At the end"))
	logging_info("Plot:                {}".format(arguments.plot))

	# register exit handler
	atexit.register(exit_handler, run)

	should_continue = query_yes_no("Continue?", default="yes")
	if not should_continue:
		exit()

	# determine compute device
	device = torch.device(arguments.device)

	# initialize model
	model = VGG19(input_channels=3)
	vgg_19_path = get_full_path("models", "vgg_19_imagenet", "vgg_19.minimum.pkl")
	if not os.path.isfile(vgg_19_path):
		logging_error("Please download the weights and biases of the VGG 19 network from " \
					  "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz, extract the archive and run the Python script "\
					  "`extract_vgg_19_weights.py`.")
	model.initialize(vgg_19_path)
	model.to(device)

	# setup extraction layer lists
	extract_layers = arguments.style_layers
	extract_layers.extend(arguments.content_layers)

	# load input data
	input_style = image.load(arguments.input_style_file, device=device)
	input_map = image.load(arguments.input_map_file, device=device)
	output_map = image.load(arguments.output_map_file, device=device)
	if arguments.output_content_file is not None:
		output_content = image.load(arguments.output_content_file, device=device)
	else:
		output_content = None

	# remove alpha channel if present
	input_style = input_style[:, :3, :, :]
	input_map = input_map[:, :3, :, :]
	output_map = output_map[:, :3, :, :]
	if output_content is not None:
		output_content = output_content[:, :3, :, :]

	# setup output image
	target = torch.autograd.Variable(torch.zeros_like(input_style), requires_grad=True)

	# perform forward pass of model to extract response for loss
	style_response = model.forward(input_style, extract_layers=arguments.style_layers)
	if output_content is not None:
		content_response = model.forward(output_content, extract_layers=arguments.content_layers)
	else:
		content_response = None

	# initialize loss
	style_loss = StyleLoss(
		style_response,
		input_map,
		output_map,
		arguments.style_layers,
		arguments.map_channel_weight,
		stride = 1
	)
	if content_response is not None:
		content_loss = ContentLoss(
			content_response,
			arguments.content_layers
		)
	else:
		content_loss = None

	# setup optimizer
	optimizer = torch.optim.LBFGS([target], lr=1.0, history_size=100)

	# setup live plot
	if arguments.save_interval is not None and arguments.plot:
		_, _, height, width = input_style.size()
		live_plot = plot.LivePlot(width, height)

	logging_info("Start generating the image.")

	# main loop
	t0 = time.time()
	for phase in range(1, arguments.num_phases + 1):
		optimizer.zero_grad()

		# compute activations from model
		activations = model.forward(target, extract_layers=extract_layers)

		# calculate loss
		loss = arguments.style_weight * style_loss.loss(activations)
		if content_loss is not None:
			loss += arguments.content_weight * content_loss.loss(activations)

		loss.backward(retain_graph=True)
		loss_val = optimizer.step(lambda: loss)

		# compute loss for output
		print_loss = loss_val.item() / arguments.style_weight
		print_loss /= arguments.map_channel_weight
		if content_loss is not None:
			print_loss /= arguments.content_weight

		t1 = time.time() - t0
		logging_info("Phase {:d}: Loss = {:f}, Escaped Time = {:.0f}m {:.0f}s".format(
			phase, loss_val.item(), t1 // 60 % 60, t1 % 60
		))

		# update live plot and save output
		if arguments.save_interval is not None and phase % arguments.save_interval == 0:
			# save output
			filepath = os.path.join(
				run.output_path, "output-{}.png".format(phase)
			)
			image.save(filepath, target)
			logging_info("Output image saved to '{}'.".format(filepath))

			# update live plot
			if arguments.plot:
				live_plot.update(target)

	# write output to disk
	filepath = os.path.join(run.output_path, "result.png")
	image.save(filepath, target)
	logging_info("Result image saved to '{}'.".format(filepath))
