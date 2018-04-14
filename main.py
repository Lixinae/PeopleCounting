from vxl import VxlCameraInfo, VxlVideo
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from scipy.signal import *
from scipy.ndimage import *

def amplitude_objects(frame, amplitude_min, amplitude_max, openingKernelSize=5):
    foreground = abs(frame.amplitude - background.amplitude)
    _, objects = cv2.threshold(foreground.astype(np.uint8), 0, amplitude_max - amplitude_min,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((openingKernelSize, openingKernelSize), np.uint8))
    return objects


def depth_objects(frame, depth_min, depth_max, channels=256, blurKernelSize=5, openingKernelSize=5, thresholdCoeff=4.5):
    image = ((channels - 1) * ((frame.depth - depth_min) / (depth_max - depth_min))).astype(np.uint8)
    image = cv2.medianBlur(image, blurKernelSize)
    bg = ((channels - 1) * ((background.depth - depth_min) / (depth_max - depth_min))).astype(np.uint8)
    bg = cv2.medianBlur(bg, blurKernelSize)
    foreground = bg.astype(np.int16) - image.astype(np.int16)
    foreground = cv2.medianBlur(foreground, blurKernelSize)
    objects = (foreground > (thresholdCoeff / 100) * channels).astype(np.uint8)
    objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((openingKernelSize, openingKernelSize), np.uint8))
    return objects


def binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max, channels=256, kernelSize=5):
    amplitude = amplitude_objects(frame, amplitude_min, amplitude_max, openingKernelSize=kernelSize)
    depth = depth_objects(frame, depth_min, depth_max, channels=channels, blurKernelSize=kernelSize,
                          openingKernelSize=kernelSize)
    return np.logical_or(amplitude, depth)


# from scipy.signal import *
from scipy.ndimage import filters


def testsum(binary_frame, sigma=3.0):
    vertical_sum = binary_frame.sum(axis=0)
    vertical_sum = filters.gaussian_filter1d(vertical_sum, sigma)
    return vertical_sum


def count_people(frame, amplitude_min, amplitude_max, depth_min, depth_max):
    """objects = binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max)


    sigma = 3.0
    vertical_sum = objects.sum(axis=0)
    vertical_sum = filters.gaussian_filter1d(vertical_sum, sigma)"""

    return 0


cameraInfo = VxlCameraInfo('OPT8241')

background = VxlVideo.readAsAvgImage("videos/video_1_ap.vxl", cameraInfo)

video = VxlVideo.read("videos/video_1.vxl", cameraInfo)

amplitude_min = min(min([frame.amplitude.min() for frame in video]), background.amplitude.min())
amplitude_max = max(max([frame.amplitude.max() for frame in video]), background.amplitude.max())
depth_min = min(min([frame.depth.min() for frame in video]), background.depth.min())
depth_max = max(max([frame.depth.max() for frame in video]), background.depth.max())

for frame in video:

    binary_frame = binary_image(frame, amplitude_min, amplitude_max, depth_min, depth_max)

    plt.figure(1)
    plt.clf()
    plt.imshow(binary_frame, cmap=plt.cm.Greys_r)
    # plt.imshow(frame.amplitude, cmap=plt.cm.Greys_r)
    # plt.imshow(frame.depth, cmap=plt.cm.Greys_r)
    nb_people = count_people(frame, amplitude_min, amplitude_max, depth_min, depth_max)
    plt.text(15, 23, 'people : ' + str(nb_people), bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    plt.axis('off')

    plt.figure(2)
    plt.clf()
    som = testsum(binary_frame)
    y = np.array(som)
    maxm = argrelmax(y)
    plt.plot(som)
    for i in maxm:
        plt.plot(i, som[i], 'o', color="red")

    plt.pause(0.001)

plt.show()

i = 10
i = 150
plt.clf()
plt.imshow(process(video.avgFrame(i, 1)), cmap=plt.cm.Greys_r)
# plt.imshow(video.avgFrame(i,1), cmap=plt.cm.Greys_r)
nb_people = "..."
plt.text(15, 23, 'people : ' + str(nb_people), bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
plt.axis('off')
plt.pause(0.001)
plt.show()

"""
for frame in video:
	plt.clf()
	plt.imshow(frame.depth, cmap=plt.cm.Greys_r)
	plt.pause(0.001)
plt.show()
"""

# foreground = abs(cv2.GaussianBlur(image1,(5,5),0) - cv2.GaussianBlur(background,(5,5),0))
# foreground = abs(image1 - background)
# foreground = cv2.GaussianBlur(abs(image1 - background),(5,5),0)
# foreground = cv2.GaussianBlur(abs(cv2.GaussianBlur(image1,(5,5),0) - cv2.GaussianBlur(background,(5,5),0)),(5,5),0)


foreground = cv2.GaussianBlur(background, (5, 5), 0) - cv2.GaussianBlur(image1, (5, 5), 0)
foreground = cv2.GaussianBlur(foreground, (5, 5), 0)
# foreground = cv2.GaussianBlur(np.maximum(foreground,0),(5,5),0)


"""
radius = 15
selem = disk(radius)
local_otsu = rank.otsu(img, selem)
"""

mini = foreground.min()
maxi = foreground.max()
forg = (255 * ((foreground - mini) / (maxi - mini))).astype(np.uint8)

_, objects = cv2.threshold(forg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# objects = cv2.adaptiveThreshold(forg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 1)


# objects = (foreground > 0).astype(np.uint8)

# bjects = (cv2.GaussianBlur(image1,(5,5),0)  < cv2.GaussianBlur(background,(5,5),0) - 0.1).astype(np.uint8)

# objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
# objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

"""
plt.imshow(background, cmap=plt.cm.Greys_r)
plt.show()

plt.imshow(image1, cmap=plt.cm.Greys_r)
plt.show()

plt.imshow(foreground, cmap=plt.cm.Greys_r)
plt.show()

plt.imshow(forg, cmap=plt.cm.Greys_r)
plt.show()

plt.imshow(objects, cmap=plt.cm.Greys_r)
plt.show()
"""
