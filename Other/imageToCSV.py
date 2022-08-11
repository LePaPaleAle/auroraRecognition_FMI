import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat, ImageEnhance
import glob
import colorsys


def cropTrees(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    mask = mask == 255
    image[mask] = (0, 0, 0)

    ind = np.where(~mask)
    # find edges of mask to crop image (so we don't need to process all the black and text pixels around the actual image)
    # amin = find minimum along an axis
    min_y = np.amin(ind[0]) - 1
    max_y = np.amax(ind[0]) + 2
    min_x = np.amin(ind[1]) - 1
    max_x = np.amax(ind[1]) + 2

    # image[mask] = (np.nan, np.nan, np.nan)

    # crop image and mask to cut away unwanted pixels
    image = image[min_y:max_y, min_x:max_x, :]
    mask = mask[min_y:max_y, min_x:max_x]

    return image, mask


CSV_PATH = '/home/moisio/Documents/auroraRecognition/samples/aurora/greenAurora/test.csv'
csv_file = open(CSV_PATH, 'w+')
filewriter = csv.writer(csv_file)
filewriter.writerow(['R', 'G', 'B', 'H', 'S', 'L', 'R - G', 'R - B', 'G - B', 'S - L'])
treeMask = Image.open('/home/moisio/Documents/auroraRecognition/mask.png')

inputImage = '/home/moisio/Documents/auroraRecognition/samples/aurora/greenAurora/06112021_225920-0004.png'
img = Image.open(inputImage)
# img, mask = cropTrees(img, treeMask)
numpyImage = np.array(img)
R, G, B = numpyImage[:, :, 0] / 255., numpyImage[:, :, 1] / 255., numpyImage[:, :, 2] / 255.
R, G, B = R.flatten(), G.flatten(), B.flatten()
# R = np.around(R[np.invert(mask)], decimals=2)
# G = np.around(G[np.invert(mask)], decimals=2)
# B = np.around(B[np.invert(mask)], decimals=2)
H, S, L = np.copy(R), np.copy(G), np.copy(B)

for idx, r in np.ndenumerate(R):
    g, b = G[idx], B[idx]
    H[idx], L[idx], S[idx] = colorsys.rgb_to_hls(r, g, b)

# H, S, L = np.around(H, decimals=2), np.around(S, decimals=2), np.around(L, decimals=2)

R, G, B = np.expand_dims(R, axis=1), np.expand_dims(G, axis=1), np.expand_dims(B, axis=1)
H, S, L = np.expand_dims(H, axis=1), np.expand_dims(S, axis=1), np.expand_dims(L, axis=1)
colorArray = rgb = np.hstack((R, G, B, H, S, L, (R - G), (R - B), (G - B), (S - L)))
# colorArray.tofile(CSV_PATH, sep=',', format='%1.2f')
np.savetxt(CSV_PATH, colorArray, delimiter=',', fmt='%.2f')
