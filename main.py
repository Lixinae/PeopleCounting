from vxl import VxlCameraInfo, VxlVideo
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from numpy.linalg import norm

def amplitude_objects(frame, amplitude_min, amplitude_max, openingKernelSize=5):
	foreground = abs(frame.amplitude - background.amplitude)
	_,objects = cv2.threshold(foreground.astype(np.uint8), 0, amplitude_max - amplitude_min, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((openingKernelSize, openingKernelSize),np.uint8))
	return objects

def depth_objects(frame, depth_min, depth_max, channels=256, blurKernelSize=5, openingKernelSize=5, thresholdCoeff=4.5):
	image = ((channels - 1)*((frame.depth - depth_min)/(depth_max - depth_min))).astype(np.uint8)
	image = cv2.medianBlur(image, blurKernelSize)	
	bg = ((channels - 1)*((background.depth - depth_min)/(depth_max - depth_min))).astype(np.uint8)
	bg = cv2.medianBlur(bg, blurKernelSize)
	foreground = bg.astype(np.int16) - image.astype(np.int16)
	foreground = cv2.medianBlur(foreground, blurKernelSize)
	objects = (foreground > (thresholdCoeff/100)*channels).astype(np.uint8)
	objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((openingKernelSize,openingKernelSize),np.uint8))
	return objects

def binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max, channels=256, kernelSize=5):
	amplitude = amplitude_objects(frame, amplitude_min, amplitude_max, openingKernelSize=kernelSize)
	depth = depth_objects(frame, depth_min, depth_max, channels=channels, blurKernelSize=kernelSize, openingKernelSize=kernelSize)
	return np.logical_or(amplitude, depth)

def compute_feature_points(vertical_sum):
	return [
		np.array([x,vertical_sum[x]]) for x in range(0,len(vertical_sum),5)
	]

def compute_shape_descriptors(binary_frame, sigma=1.5, threshold=0.):
	height,width = binary_frame.shape
	vertical_sum = binary_frame.sum(axis=0)
	vertical_sum = filters.gaussian_filter1d(vertical_sum, (sigma/100)*width)
	descriptors = []
	current_descriptor = []
	for value in vertical_sum:
		if value > threshold:
			current_descriptor.append(value)
		elif current_descriptor != []:
			descriptors.append(compute_feature_points(current_descriptor))
			current_descriptor = []
	return vertical_sum, descriptors

def is_people(shape_descriptor, minRatio=0.45, maxRatio=0.55):
	n = len(shape_descriptor)
	if n < 2:
		return False
	width = norm(shape_descriptor[-1]-shape_descriptor[0])
	height = shape_descriptor[n//2][0]
	ratio = height/width
	return (minRatio <= ratio <= maxRatio)

def count_people(shape_descriptors):
	return len([shape_descriptor for shape_descriptor in shape_descriptors if is_people(shape_descriptor)])


cameraInfo = VxlCameraInfo('OPT8241')

background = VxlVideo.readAsAvgImage("videos/video_1_ap.vxl", cameraInfo)

video = VxlVideo.read("videos/video_1.vxl", cameraInfo)

amplitude_min = min(min([frame.amplitude.min() for frame in video]), background.amplitude.min())
amplitude_max = max(max([frame.amplitude.max() for frame in video]), background.amplitude.max())
depth_min = min(min([frame.depth.min() for frame in video]),background.depth.min())
depth_max = max(max([frame.depth.max() for frame in video]),background.depth.max())

for frame in video:
	
	binary_frame = binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max)
	vertical_sum,shape_descriptors = compute_shape_descriptors(binary_frame)
	nb_people = count_people(shape_descriptors)
	
	plt.figure(1)
	plt.clf()
	plt.imshow(binary_frame, cmap=plt.cm.Greys_r)
	#plt.imshow(frame.amplitude, cmap=plt.cm.Greys_r)
	#plt.imshow(frame.depth, cmap=plt.cm.Greys_r)
	plt.text(15, 23, 'people : ' + str(nb_people), bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.axis('off')
	
	plt.figure(2)
	plt.clf()
	plt.xticks(range(0,350,40))
	plt.yticks(range(0,300,40))
	plt.plot(vertical_sum)
	
	plt.pause(0.001)
