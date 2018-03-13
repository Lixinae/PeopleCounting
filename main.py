from vxl import VxlCameraInfo, VxlVideo
import numpy as np
import cv2
import matplotlib.pyplot as plt

cameraInfo = VxlCameraInfo('OPT8241')

video = VxlVideo.read("videos/test2.vxl", cameraInfo)

for frame in video:
	plt.clf()
	plt.imshow(frame.amplitude, cmap=plt.cm.Greys_r)
	plt.pause(0.001)
plt.show()
