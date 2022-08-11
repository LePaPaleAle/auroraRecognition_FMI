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

samplePath = r'/home/moisio/Documents/auroraRecognition/samples3/no-aurora'
# sampleFolders = ['/home/moisio/Documents/auroraRecognition/samples2/aurora/allAurora/18102021_021541-0021.png']
# folderNames = ['Testing']
sampleFolders = sorted([os.path.join(samplePath, file) for file in os.listdir(samplePath)])
folderNames = sorted(os.listdir(samplePath))
for folder, name in zip(sampleFolders, folderNames):
    imagePaths = sorted(glob.glob(folder + '/*.png'))
    imageCount = len(imagePaths)

    mean, median, stddev, l_avg, l_std, r_g, r_b, g_b, r_min_g, r_min_b, g_min_b = [], [], [], [], [], [], [], [], [], [], []

    for inputImagePath in imagePaths:
        pilImage = Image.open(inputImagePath)
        numpyImage = np.array(pilImage)
        R, G, B = numpyImage[:, :, 0], numpyImage[:, :, 1], numpyImage[:, :, 2]
        imageStatsRGB = ImageStat.Stat(pilImage)
        imageMean = imageStatsRGB.mean
        imageMedian = imageStatsRGB.median
        imageStddev = imageStatsRGB.stddev
        imageMean_percentage = np.array(imageMean) / 255.
        imageMedian_percentage = np.array(imageMedian) / 255.
        imageStddev_percentage = np.array(imageStddev) / 255.
        l_mean, l_stddev = getImage_L(pilImage)
        r_g_avg = imageMean_percentage[0] / imageMean_percentage[1]
        r_b_avg = imageMean_percentage[0] / imageMean_percentage[2]
        g_b_avg = imageMean_percentage[1] / imageMean_percentage[2]
        r_min_g_avg = imageMean_percentage[0] - imageMean_percentage[1]
        r_min_b_avg = imageMean_percentage[0] - imageMean_percentage[2]
        g_min_b_avg = imageMean_percentage[1] - imageMean_percentage[2]
        mean.append(imageMean_percentage), median.append(imageMedian_percentage), stddev.append(imageStddev_percentage),
        l_avg.append(l_mean), l_std.append(l_stddev), r_g.append(r_g_avg), r_b.append(r_b_avg), g_b.append(g_b_avg),
        r_min_g.append(r_min_g_avg), r_min_b.append(r_min_b_avg), g_min_b.append(g_min_b_avg)

    r_min_g_stddev = np.std(np.array(r_min_g))
    r_min_b_stddev = np.std(np.array(r_min_b))
    g_min_b_stddev = np.std(np.array(g_min_b))
    print()
    print(name)
    print('L:', sum(l_avg) / imageCount)
    print('L standard deviation:', sum(l_std) / imageCount)
    print('Image RGB mean:', sum(mean) / imageCount)
    print('Image RGB median:', sum(median) / imageCount)
    print('Image RGB standard deviation:', sum(stddev) / imageCount)
    print('\nImage r / g:', sum(r_g) / imageCount,
          '\nImage r / b:', sum(r_b) / imageCount,
          '\nImage g / b:', sum(g_b) / imageCount,
          '\nImage r - g:', sum(r_min_g) / imageCount,
          '\nImage r - b:', sum(r_min_b) / imageCount,
          '\nImage g - b:', sum(g_min_b) / imageCount,
          '\nImage r - g stddev:', r_min_g_stddev,
          '\nImage r - b stddev:', r_min_g_stddev,
          '\nImage g - b stddev:', g_min_b_stddev,
          '\n')


