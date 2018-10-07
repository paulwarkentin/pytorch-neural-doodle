##
## pytorch-neural-doodle/src/utils/plot.py
##
## Created by Bastian Boll <mail@bbboll.com> on 01/09/2018.
## Updated by Bastian Boll <mail@bbboll.com> on 01/09/2018.
##

import matplotlib.pyplot as plt
import numpy as np


class LivePlot(object):
	"""Continuously updating plot of the target image.
	"""

	def __init__(self, w, h):
		super().__init__()

		self.width = w
		self.height = h
		self.image = np.zeros((h,w,3), dtype='uint8')

		self.figure = plt.figure()
		self.ax = self.figure.add_subplot(111)
		self.display_img = self.ax.imshow(self.image)
		plt.ion()
		plt.show()


	def update(self, img):
		""""""
		self.image = img.detach().cpu().numpy()
		self.image = self.image.reshape((3, self.height, self.width)).transpose((1,2,0))
		self.image = (self.image + 1.0)*(256/2)
		self.image = self.image.astype('uint8')
		self.display_img.set_data(self.image)
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()
