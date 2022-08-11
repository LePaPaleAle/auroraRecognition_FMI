import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat, ImageEnhance
import glob
# read test image and create an interactive window
def getMousePosition(event, x, y, flags, params):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y

def getImage_L(inputPILImage):
    npImg = np.array(inputPILImage)
    r = npImg[:, :, 0] / 255.
    g = npImg[:, :, 1] / 255.
    b = npImg[:, :, 2] / 255.
    minimum = np.minimum(np.minimum(r, g), b)
    maximum = np.maximum(np.maximum(r, g), b)
    L = (minimum + maximum) / 2
    L_mean = np.mean(L)
    L_stddev = np.std(L)

    return L_mean, L_stddev


def get9pxAreaRGBvalues(x, y, srcImage):
    red, green, blue = [], [], []
    area_9_px = srcImage[y - 1:y + 2, x - 1:x + 2]
    for rgb in area_9_px:
        for colorvalues in rgb:
            r, g, b = colorvalues
            red.append(r), green.append(g), blue.append(b)
    return red, green, blue


inputImages = ['/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/difficult_2/17092021_013655-0012.png']
#                '/home/moisio/Documents/auroraRecognition/samples3/aurora/aurora3/06112021_165543-0012.png']
# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/samples3/no-aurora/no-aurora13/*.png')
# inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/difficult_2/*.png'))
for inputImage in inputImages:
    pilImage = Image.open(inputImage)
    numpyImage = np.array(pilImage)
    R, G, B = numpyImage[:, :, 0], numpyImage[:, :, 1], numpyImage[:, :, 2]
    R_hist = np.histogram(R, bins=256, range=[0, 255])
    G_hist = np.histogram(G, bins=256, range=[0, 255])
    B_hist = np.histogram(B, bins=256, range=[0, 255])
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(R_hist[0], color='r')
    ax[1].plot(G_hist[0], color='g')
    ax[2].plot(B_hist[0], color='b')

    imageStatsRGB = ImageStat.Stat(pilImage)
    imageMean = np.array(imageStatsRGB.mean) / 255.
    imageMedian = np.array(imageStatsRGB.median) / 255.
    imageStddev = np.array(imageStatsRGB.stddev) / 255.
    imageMean_percentage = np.array(imageMean) / 255.
    imageMedian_percentage = np.array(imageMedian) / 255.
    imageStddev_percentage = np.array(imageStddev) / 255.
    l_mean, l_stddev = getImage_L(pilImage)
    r_g = (R / G)
    r_b = (R / B)
    g_b = (G / B)
    r_min_g = R / 255. - G / 255.
    r_min_b = R / 255. - B / 255.
    g_min_b = G / 255. - B / 255.
    r_min_g_mean = np.mean(r_min_g)
    r_min_b_mean = np.mean(r_min_b)
    g_min_b_mean = np.mean(g_min_b)
    r_min_g_stddev = np.std(r_min_g)
    r_min_b_stddev = np.std(r_min_b)
    g_min_b_stddev = np.std(g_min_b)

    r_g_mean = np.mean(r_g)
    r_b_mean = np.mean(r_b)
    g_b_mean = np.mean(g_b)
    r_g_stddev = np.std(r_g)
    r_b_stddev = np.std(r_b)
    g_b_stddev = np.std(g_b)
    print('L:', l_mean)
    print('L standard deviation:', l_stddev)
    print('Image RGB mean:', imageMean)
    print('Image RGB median:', imageMedian)
    print('Image RGB standard deviation:', imageStddev)
    print('\nImage r / g mean:', r_g_mean,
          '\nImage r / b mean:', r_b_mean,
          '\nImage g / b mean:', g_b_mean,
          '\nImage r / g stddev:', r_g_stddev,
          '\nImage r / b stddev:', r_b_stddev,
          '\nImage g / b stddev:', g_b_stddev,
          '\nImage r - g mean:', r_min_g_mean,
          '\nImage r - b mean:', r_min_b_mean,
          '\nImage g - b mean:', g_min_b_mean,
          '\nImage r - g stddev:', r_min_g_stddev,
          '\nImage r - b stddev:', r_min_g_stddev,
          '\nImage g - b stddev:', g_min_b_stddev,
          '\n')
    plt.show()


