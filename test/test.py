import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import *
from scipy.signal import *
from scipy.ndimage import *
import scipy


# Pure bullshit
# def testMaxLoc(som, l):
#     maximum = []
#     somClean = [x for x in som if x > 0]
#     size = len(somClean)
#     print(somClean)
#     offset = 0
#     for i in range(len(som)):
#         if som[i] != 0:
#             offset = i
#             break
#
#     for i in range(size):
#         max = 0
#         indice = 0
#         if i-l > 0 and i+l < size:
#             for j in range(i - l, i + l):
#                 if somClean[j] > max:
#                     max = somClean[j]
#                     indice = j
#         maximum.append(indice+offset)
#     return maximum


def isPerson(img, sigma):
    som = []
    h, w = img.shape
    for i in range(w):
        tmpSum = 0
        for j in range(h):
            tmpSum += img[j][i]
        som.append(tmpSum)

    x = np.array(range(w))
    filtered = filters.gaussian_filter1d(som, sigma)
    y = np.array(filtered)

    maxm = argrelmax(y)  # maxm = argrelextrema(y, np.greater)
    minm = argrelextrema(y, np.less)

    plt.plot(x, y)

    for i in maxm:
        plt.plot(i, filtered[i], 'o', color="red")
        # plt.annotate('Max Local',
        #          ha='center', va='bottom',
        #          xytext=(-1.5 +i, 3. +som[i]),
        #          xy=(i, som[i]),
        #          arrowprops={'facecolor': 'black', 'shrink': 0.05})
    plt.show()  # affiche la figure a l'ecran
    isMiddleMax = filtered[maxm[0][1]] > filtered[maxm[0][0]] and filtered[maxm[0][1]] > filtered[maxm[0][2]]
    return len(maxm[0]) == 3 and isMiddleMax


img = cv2.imread("./personne.png", 0)
img3 = cv2.imread("./personne2.png", 0)
img2 = cv2.imread("./salesman2.png", 0)
img2 = cv2.bitwise_not(img2)
print(isPerson(img, 3))
print(isPerson(img3, 3))
print("//////////////////////////////////////////")
print(isPerson(img2, 3))

# cv2.imshow('image', img3)
# cv2.waitKey(0)
# img3 = cv2.GaussianBlur(img,(5,5),1)
# cv2.imshow('image', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
