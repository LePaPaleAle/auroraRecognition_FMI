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

    classification_mask = (L - S >= 0.5)
    # 0.3 <= L <= 0.9 is good

    coloredImage[classification_mask] = colors[0]
    classifiedImage[classification_mask] = 1
    handles.append(patches.Patch(color=colors[0] / 255., label='Aurora'))

    return coloredImage, classifiedImage, handles


# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/difficult/*.png')
# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/clear_aurora/gray aurora/*.png')
inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/greenAurora/*.png')
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
    path.append('testing05082022')
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










