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


CSV_PATH = '/home/moisio/Documents/auroraRecognition/testSetLarge/noAurora/test_csv'
csv_file = open(CSV_PATH, 'w+')
filewriter = csv.writer(csv_file)
filewriter.writerow(['R', 'G', 'B', 'H', 'S', 'L', 'R - G', 'R - B', 'G - B', 'S - L'])
treeMask = Image.open('/home/moisio/Documents/auroraRecognition/MUO_mask.png')
# inputImages = ['/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/difficult_2/17092021_013655-0012.png']
# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/greenAurora/*.png')
inputImages = glob.glob('/home/moisio/MUO/dataSet/yellow/*.png')
for inputImage in inputImages:
    img = Image.open(inputImage)
    img, mask = cropTrees(img, treeMask)
    numpyImage = np.array(img)
    R, G, B = numpyImage[:, :, 0] / 255., numpyImage[:, :, 1] / 255., numpyImage[:, :, 2] / 255.
    # R, G, B = R.flatten(), G.flatten(), B.flatten()
    R = R[np.invert(mask)]
    G = G[np.invert(mask)]
    B = B[np.invert(mask)]
    H, S, L = np.copy(R), np.copy(G), np.copy(B)

    for idx, r in np.ndenumerate(R):
        g, b = G[idx], B[idx]
        H[idx], L[idx], S[idx] = colorsys.rgb_to_hls(r, g, b)

    R, G, B = np.expand_dims(R, axis=1), np.expand_dims(G, axis=1), np.expand_dims(B, axis=1)
    H, S, L = np.expand_dims(H, axis=1), np.expand_dims(S, axis=1), np.expand_dims(L, axis=1)
    colorArray = rgb = np.hstack((R, G, B, H, S, L, (R - G), (R - B), (G - B), (S - L)))
    colorArray.tofile(CSV_PATH, sep=',')


    R_hist = np.histogram(R, bins=256, range=(0.0, 1.0))
    G_hist = np.histogram(G, bins=256, range=[0, 1])
    B_hist = np.histogram(B, bins=256, range=[0, 1])
    H_hist = np.histogram(H, bins=360, range=[0.0, 1.0])
    S_hist = np.histogram(S, bins=256, range=[0.0, 1.0])
    L_hist = np.histogram(L, bins=256, range=[0.0, 1.0])
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    ax[0, 0].imshow(Image.fromarray(numpyImage))
    ax[0, 1].plot(R_hist[0], color='r')
    ax[0, 1].set_title('R')
    ax[0, 2].plot(G_hist[0], color='g')
    ax[0, 2].set_title('G')
    ax[0, 3].plot(B_hist[0], color='b')
    ax[0, 3].set_title('B')

    ax[1, 0].hist(H, bins=50, color='c')
    ax[1, 0].set_title('H')
    ax[1, 1].hist(S, bins=50, color='m')
    ax[1, 1].set_title('S')
    ax[1, 2].hist(L, bins=50, color='y')
    ax[1, 2].set_title('L')
    ax[1, 3].hist(S - L, bins=50, color='k')
    ax[1, 3].set_title('S - L')

    ax[2, 0].hist(R - G, bins=50, color='k')
    ax[2, 0].set_title('R - G')
    ax[2, 1].hist(R - B, bins=50, color='k')
    ax[2, 1].set_title('R - B')
    ax[2, 2].hist(G - B, bins=50, color='k')
    ax[2, 2].set_title('G - B')
    ax[2, 3].hist(R - (G + B), bins=50, color='k')
    ax[2, 3].set_title('R - (G + B)')

    plt.show()
