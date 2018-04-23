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
	objects = (foreground > (thresholdCoeff/100)*channels).astype(np.uint8)
	objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((openingKernelSize,openingKernelSize),np.uint8))
	return objects

def binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max, channels=256, kernelSize=5):
	amplitude = amplitude_objects(frame, amplitude_min, amplitude_max, openingKernelSize=kernelSize)
	depth = depth_objects(frame, depth_min, depth_max, channels=channels, blurKernelSize=kernelSize, openingKernelSize=kernelSize)
	return np.logical_or(amplitude, depth)

def compute_components(function, threshold):
	components = []
	current_component = []
	for i in range(len(function)):
		if function[i] > threshold:
			current_component.append(i)
		elif current_component != []:
			components.append(np.array(current_component))
			current_component = []
	return components

def compute_feature_points(vertical_sum, gradient, threshold=1.5, min_points=2):
	components = compute_components(gradient, threshold)
	first = np.array([0, vertical_sum[0]])
	last = np.array([len(vertical_sum)-1, vertical_sum[len(vertical_sum)-1]])
	return np.array([first] + [
		np.array([c[len(c)//2], vertical_sum[c[len(c)//2]]]) 
		for c in components
		if len(c) > min_points
	] + [last])

def compute_shape_descriptors(binary_frame, sigma=1.5, threshold=0.):
	height,width = binary_frame.shape
	vertical_sum = binary_frame.sum(axis=0)
	vertical_sum = filters.gaussian_filter1d(vertical_sum, (sigma/100)*width)
	gradient = np.array([abs(vertical_sum[i+1] - vertical_sum[i]) for i in range(len(vertical_sum)-1)])
	components = compute_components(vertical_sum, threshold)
	return map(lambda i : compute_feature_points(vertical_sum[i], gradient[i]), components)

def is_people(shape_descriptor, minRatio=1.25, maxRatio=1.75):
	n = len(shape_descriptor)	
	if n < 3:
		return False
	width = norm(shape_descriptor[-1]-shape_descriptor[0])
	if width < 1:
		return False
	height = np.max(shape_descriptor[n//2 - 1 : n//2 + 2, 1])
	ratio = height/width
	return (minRatio <= ratio <= maxRatio)

def count_people(frame, amplitude_min, amplitude_max, depth_min, depth_max):
	binary_frame = binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max)
	shape_descriptors = compute_shape_descriptors(binary_frame)
	return len([shape_descriptor for shape_descriptor in shape_descriptors if is_people(shape_descriptor)])

cameraInfo = VxlCameraInfo('OPT8241')

background = VxlVideo.readAsAvgImage("videos/test_27_3_ap_short_range.vxl", cameraInfo)

video = VxlVideo.read("videos/test_27_3_1p_short_range.vxl", cameraInfo)

amplitude_min = min(min([frame.amplitude.min() for frame in video]), background.amplitude.min())
amplitude_max = max(max([frame.amplitude.max() for frame in video]), background.amplitude.max())
depth_min = min(min([frame.depth.min() for frame in video]),background.depth.min())
depth_max = max(max([frame.depth.max() for frame in video]),background.depth.max())

for frame in video:
	
	nb_people = count_people(frame, amplitude_min, amplitude_max, depth_min, depth_max)

	plt.clf()
	plt.imshow(frame.amplitude, cmap=plt.cm.Greys_r)
	plt.text(15, 23, 'people : ' + str(nb_people), bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
	plt.axis('off')
	plt.pause(0.001)

