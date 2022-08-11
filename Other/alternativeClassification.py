from PIL import Image
import numpy as np
from matplotlib import pyplot as plt, patches
import glob
import os
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


def classifyImage(np_img_array):
    height, width, _ = np.shape(np_img_array)
    handles = []
    coloredImage = np.copy(np_img_array)
    classifiedImage = np.full_like(np.zeros((height, width)), False, dtype=bool)
    cmap = plt.cm.get_cmap('RdYlGn', 6)
    colors = cmap(range(6))[:, :3] * 255
    R, G, B = np_img_array[:, :, 0] / 255., np_img_array[:, :, 1] / 255., np_img_array[:, :, 2] / 255.
    H, S, L = np.copy(R), np.copy(G), np.copy(B)

    for idx, r in np.ndenumerate(R):
        g, b = G[idx], B[idx]
        H[idx], L[idx], S[idx] = colorsys.rgb_to_hls(r, g, b)

    # minimum = np.minimum(np.minimum(R, G), B)
    # maximum = np.maximum(np.maximum(R, G), B)
    # L = (minimum + maximum) / 2

    green_mask = (0.15 <= H) & (H <= 0.3) & (S >= 0.15) & (0.3 <= L) & (L <= 0.95) & \
                 (-0.5 <= (S - L)) & ((S - L) <= 0.1) & (0.05 <= (G - R)) & (0.1 <= (G - B)) & (0.05 <= (R - B)) & \
                 ((R - B) <= 0.25) & (-0.5 <= (G - (B + R))) & ((G - (B + R)) <= 0.2)

    red_mask = ((0. <= H) & (H <= 0.2) | (0.85 <= H)) & (S >= 0.2) & (0.2 <= L) & (L <= 0.95) & \
               (-0.4 <= (S - L)) & ((S - L) <= 0.15) & (0.1 <= (R - G)) & ((R - G) <= 0.3) & (0.1 <= (R - B)) & \
               ((R - B) <= 0.4) & (-0.1 <= (G - B)) & ((G - B) <= 0.3) & (-0.7 <= (R - (G + B))) & ((R - (G + B)) <= 0)

    # red_mask = (R > B) & (R > G) & (L >= 0.3) & (L <= 0.95) & (R >= 0.35) & \
               # ((R - G) >= 0.1) & ((G - B) >= -0.035) & ((G - B) <= 0.035)

    # ((G + B) <= 1.2) & ((R - G) >= 0.2) &

    purple_mask = (0.65 <= H) & (H <= 1) & (S <= 0.8) & (0.3 <= L) & (L <= 0.95) & \
                  (-0.6 <= (S - L)) & ((S - L) <= 0.) & (-0.1 <= (R - G)) & ((R - G) <= 0.1) & (0. <= (B - G)) & \
                  ((B - G) <= 0.3) & (-0.05 <= (B - R)) & ((B - R) <= 0.35) & \
                  (-0.8 <= (B - (R + G))) & ((B - (R + G)) <= -0.1)

    dark_sky_mask = (0. <= H) & (H <= 0.2) & (S <= 0.2) & (L <= 0.2) & \
                    (-0.2 <= (S - L)) & ((S - L) <= 0.2) & (-0.05 <= (R - G)) & ((R - G) <= 0.05) & \
                    (-0.1 <= (R - B)) & ((R - B) <= 0.1) & (-0.1 <= (G - B)) & ((G - B) <= 0.1) & \
                    (R <= 0.2) & (G <= 0.2) & (B <= 0.2)

    dark_clouds_mask = (0.1 <= S) & (S <= 0.5) & (0.1 <= S) & (S <= 0.5) & (-0.1 <= (R - G)) & ((R - G) <= 0.1) & \
                       (-0.1 <= (R - B)) & ((R - B) <= 0.1) & (-0.1 <= (G - B)) & ((G - B) <= 0.1)

    # cloud_mask = (L < 0.35) & (S < 0.3)

    # green_mask = (R <= G) & (R >= B) & (G >= B) & (L > 0.4) & (L <= 0.98) & \
    #              ((R - G) < -0.01) & ((R - B) > 0.01) & ((G - B) > 0.01)
    # cyan_mask = (R < G) & (R < B) & (G < B) & (L > 0.4) & (L <= 0.98)
                # ((R - G) < -0.01) & ((R - B) < -0.01) & ((G - B) > 0.01)
    # red_mask_1 = (R >= G) & (R >= B) & (G >= B) & (L >= 0.45) & (L <= 0.85) & \
    #              ((R - G) > 0.01) & (R - B > 0.18) & ((G - B) > 0.018)
    # red_mask_2 = (R >= G) & (R >= B) & (G <= B) & (L >= 0.45) & (L <= 0.85) & \
    #              ((R - B) > 0.015)
    # purple_mask = (R >= G) & (R <= B) & (G <= B) & (L > 0.5) & (L <= 0.98) & \
    #               ((R - G) > 0.03) & ((R - B) < -0.12) & ((G - B) < -0.15)
    # cloud_mask = (L < 0.85) & \
    #              (((-0.01 <= (R - G)) & ((R - G) <= 0.01)) |
    #              ((-0.01 <= (R - B)) & ((R - B) <= 0.01)) |
    #              ((-0.01 <= (G - B)) & ((G - B) <= 0.01)))


        # no_aurora_mask_1 = (B > R) & (G > R) & (B > G)
    # no_aurora_mask_2 = (0.11 > (R - B)) & ((R - B) > -0.11) & \
    #                    (0.11 > (R - G)) & ((R - G) > -0.11) & \
    #                    (0.11 > (G - B)) & ((G - B) > -0.11) & \
    #                    (L < 0.99)
    #
    # green_mask = (G > R) & (G > B) & ((R - G) < -0.03)
    # red_mask = (R > G) & (R > B) & ((R - G) > 0.1) & ((R - B) > 0.1)
    # purple_mask = (R > G) & (B > G)

    coloredImage[green_mask] = colors[0]
    classifiedImage[green_mask] = 1
    handles.append(patches.Patch(color=colors[0] / 255., label='Green Aurora'))

    coloredImage[red_mask] = colors[1]
    classifiedImage[red_mask] = 1
    handles.append(patches.Patch(color=colors[1] / 255., label='Red Aurora'))

    coloredImage[purple_mask] = colors[2]
    classifiedImage[purple_mask] = 1
    handles.append(patches.Patch(color=colors[2] / 255., label='Purple Aurora'))

    coloredImage[dark_clouds_mask] = colors[3]
    classifiedImage[dark_clouds_mask] = 0
    handles.append(patches.Patch(color=colors[3] / 255., label='Dark Clouds'))

    coloredImage[dark_sky_mask] = colors[4]
    classifiedImage[dark_sky_mask] = 0
    handles.append(patches.Patch(color=colors[4] / 255., label='Dark Sky'))

    # coloredImage[red_mask_2] = colors[5]
    # classifiedImage[red_mask_2] = 1
    # handles.append(patches.Patch(color=colors[5] / 255., label='Red Aurora 2'))

    return coloredImage, classifiedImage, handles


# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/difficult/*.png')
# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/clear_aurora/gray aurora/*.png')
inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/*.png')
# inputImages = ['/home/moisio/Documents/auroraRecognition/testSetLarge/31012022_041552-0008.png']
treeMask = Image.open('/home/moisio/Documents/auroraRecognition/mask.png')
for image in inputImages:
    img = Image.open(image)
    img, mask = cropTrees(img, treeMask)
    R = img[:, :, 0]
    N = np.size(R[np.nonzero(R)])
    coloredImage, classifiedImage, handles = classifyImage(img)
    coloredImage[mask] = (0, 0, 0)
    coloredImage = Image.fromarray(coloredImage)
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    # original image
    im0 = ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # masked image
    im1 = ax[1].imshow(coloredImage)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)

    path = os.path.abspath(image).split('/')
    file_name = path[-1]
    path = path[:-1]
    path.append('testing_09082022')
    # check which pixels are categorised as moon,clear,aurora
    # total number of pixels in image
    auroraPixels = classifiedImage[np.invert(mask)]
    # auroraPixelCount = classifiedImage[np.nonzero(classifiedImage)].size
    auroraPixelCount = auroraPixels[auroraPixels == 1].size
    noAuroraPixelCount = auroraPixels[auroraPixels == 0].size
    # noAuroraPixelCount = classifiedImage[classifiedImage == 0].size
    # if certain number of pixels are in certain category, image is categorized as this category
    if auroraPixelCount >= 2500:
        path.append('aurora')
        ending = '_aurora.png'
    else:
        path.append('no-aurora')
        ending = '_no-aurora.png'
    path = '/'.join(path)
    plt.suptitle('Aurora: {}, No Aurora: {}'.format(auroraPixelCount, noAuroraPixelCount), color='blue')
    # if folder for category doesn't exist, create folder
    if not os.path.exists(path):
        os.makedirs(path)
    # save categorised image to category folder
    fig.savefig('/'.join([path, file_name])[:-4] + ending, bbox_inches='tight', facecolor='black')
    # plt.show()
    plt.close()










