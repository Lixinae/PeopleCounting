import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import *
from scipy.signal import *
from scipy.ndimage import *
import scipy

img = cv2.imread("./humanChienFondSeuil.png", 0)
img2 = cv2.imread("./salesman2.png", 0)
img2 = cv2.bitwise_not(img2)


def testMaxLoc(som, l):
    maximum = []
    somClean = [x for x in som if x > 0]
    size = len(somClean)
    print(somClean)
    offset = 0
    for i in range(len(som)):
        if som[i] != 0:
            offset = i
            break

    for i in range(size):
        max = 0
        indice = 0
        if i-l > 0 and i+l < size:
            for j in range(i - l, i + l):
                if somClean[j] > max:
                    max = somClean[j]
                    indice = j
        maximum.append(indice+offset)
    return maximum


def maxLoc(img):
    som = []
    h, w = img.shape
    for i in range(w):
        tmpSum = 0
        for j in range(h):
            tmpSum += img[j][i]
        som.append(tmpSum)
    # todo faire un max local avec seuil minimum
    print(som)

    x = np.array(range(w))
    filtered = filters.gaussian_filter1d(som,3)
    print(filtered)
    y = np.array(filtered)

    # maxm = argrelextrema(y, np.greater)
    # maxm = argrelmax(y)
    maxm = testMaxLoc(filtered,10)

    minm = argrelextrema(y, np.less)
    print(maxm)
    print(minm)

    # Supprime la gauche et ne conserve que le pic central
    # index = 0
    # for i in range(len(som)):
    #     if som[i] != 0:
    #         index = i
    #         break
    # for i in range(index, minm[0][0]):
    #     som[i] = 0
    #
    # # Supprime la droite
    # index2 = 0
    # for i in range(len(som) - 1, 0, -1):
    #     if som[i] != 0:
    #         index2 = i
    #         break
    # for i in range(index2, minm[0][len(minm[0]) - 1],-1):
    #     som[i] = 0
    #
    # print (index)
    #
    # print(index2)
    # sizePers = index2 - index
    # print("size :"+str(sizePers))
    # print(str(sizePers*100/w) + "%")

    # y = np.array(som)
    plt.plot(x, y)

    for i in maxm:
        plt.plot(i, filtered[i], 'o', color="red")
        # plt.annotate('Max Local',
        #          ha='center', va='bottom',
        #          xytext=(-1.5 +i, 3. +som[i]),
        #          xy=(i, som[i]),
        #          arrowprops={'facecolor': 'black', 'shrink': 0.05})
    plt.show()  # affiche la figure a l'ecran


# maxLoc(img)
print("//////////////////////////////////////////")
maxLoc(img2)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
