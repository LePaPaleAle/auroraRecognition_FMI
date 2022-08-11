from PIL import Image
import numpy as np
from matplotlib import pyplot as plt, patches
import glob
import os


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
    cmap = plt.cm.get_cmap('RdYlGn', 7)
    colors = cmap(range(7))[:, :3] * 255
    R, G, B = np_img_array[:, :, 0] / 255., np_img_array[:, :, 1] / 255., np_img_array[:, :, 2] / 255.
    minimum = np.minimum(np.minimum(R, G), B)
    maximum = np.maximum(np.maximum(R, G), B)
    L = (minimum + maximum) / 2
    ''' MUONIO CLASSIFICATION VALUES!'''
    green_mask = (R < G) & (R >= B) & (G > B) & (L > 0.015) & (L <= 0.98) & \
                 ((R - G) <= -0.01) & ((R - B) >= 0.01) & ((G - B) >= 0.01)

    cyan_mask = (R < G) & (R <= B) & (G > B) & (L > 0.15) & (L <= 0.98) & \
                ((R - G) <= -0.01) & ((R - B) <= -0.01) & ((G - B) >= 0.01)

    cloud_mask = (L <= 0.95) & \
                 (((-0.01 < (R - G)) & ((R - G) < 0.01)) |
                 ((-0.01 < (R - B)) & ((R - B) < 0.01)) |
                 ((-0.01 < (G - B)) & ((G - B) < 0.01)))

    yellow_cloud_mask = (R > G) & (R > B) & (G > B) & (L > 0.3) & \
                        ((R - G) > 0) & ((R - B) > 0.1) & ((G - B) > 0.1)

    # KEVO CLASSIFICATION VALUES!
    green_mask = (R < G) & (R >= B) & (G > B) & (L > 0.2) & (L <= 0.98) & \
                 ((R - G) < -0.01) & ((R - B) > 0.01) & ((G - B) > 0.01)

    cyan_mask = (R < G) & (R <= B) & (G > B) & (L > 0.2) & (L <= 0.98) & \
                ((R - G) < -0.01) & ((R - B) < -0.01) & ((G - B) > 0.01)

    red_mask_1 = (R > G) & (R > B) & (G >= B) & (L >= 0.45) & (L <= 0.85) & \
                 ((R - G) > 0.01) & (R - B > 0.18) & ((G - B) > 0.018)

    red_mask_2 = (R > G) & (R > B) & (G <= B) & (L >= 0.45) & (L <= 0.85) & \
                 ((R - B) > 0.015)

    purple_mask = (R >= G) & (R <= B) & (G <= B) & (L > 0.5) & (L <= 0.9) & \
                  ((R - G) > 0.03) & ((R - B) < -0.12) & ((G - B) < -0.15)

    cloud_mask = (L < 0.85) & \
                 (((-0.01 <= (R - G)) & ((R - G) <= 0.01)) |
                 ((-0.01 <= (R - B)) & ((R - B) <= 0.01)) |
                 ((-0.01 <= (G - B)) & ((G - B) <= 0.01)))

    coloredImage[green_mask] = colors[0]
    classifiedImage[green_mask] = 1
    handles.append(patches.Patch(color=colors[0] / 255., label='Green Aurora'))

    coloredImage[cyan_mask] = colors[3]
    classifiedImage[cyan_mask] = 1
    handles.append(patches.Patch(color=colors[3] / 255., label='Cyan'))

    coloredImage[red_mask_1] = colors[1]
    classifiedImage[red_mask_1] = 1
    handles.append(patches.Patch(color=colors[1] / 255., label='Red Aurora 1'))

    coloredImage[red_mask_2] = colors[5]
    classifiedImage[red_mask_2] = 1
    handles.append(patches.Patch(color=colors[5] / 255., label='Red Aurora 2'))

    coloredImage[purple_mask] = colors[2]
    classifiedImage[purple_mask] = 1
    handles.append(patches.Patch(color=colors[2] / 255., label='Purple Aurora'))

    coloredImage[cloud_mask] = colors[4]
    classifiedImage[cloud_mask] = 0
    handles.append(patches.Patch(color=colors[4] / 255., label='Cloud'))

    coloredImage[yellow_cloud_mask] = colors[6]
    classifiedImage[yellow_cloud_mask] = 0
    handles.append(patches.Patch(color=colors[6] / 255., label='Saturated Cloud'))

    return coloredImage, classifiedImage, handles


which = 'KEV'

if which == 'MUO':
    inputImages = glob.glob('/home/moisio/MUO/dataSet/*.png')
    treeMask = Image.open('/home/moisio/Documents/auroraRecognition/MUO_mask.png')
elif which == 'KEV':
    inputImages = glob.glob('/home/moisio/KEV/dataSet/*.png')
    treeMask = Image.open('/home/moisio/Documents/auroraRecognition/mask.png')
else:
    print('Choose the location!')
    exit()

# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/difficult/*.png')
# inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/experimenting/clear_aurora/gray aurora/*.png')
# inputImages = glob.glob('/home/moisio/KEV/dataSet/*.png')
# treeMask = Image.open('/home/moisio/Documents/auroraRecognition/mask.png')
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
    path.append('testing')
    # check which pixels are categorised as moon,clear,aurora
    # total number of pixels in image
    auroraPixels = classifiedImage[np.invert(mask)]
    # auroraPixelCount = classifiedImage[np.nonzero(classifiedImage)].size
    auroraPixelCount = auroraPixels[auroraPixels == 1].size
    noAuroraPixelCount = auroraPixels[auroraPixels == 0].size
    # noAuroraPixelCount = classifiedImage[classifiedImage == 0].size
    # if certain number of pixels are in certain category, image is categorized as this category
    if auroraPixelCount >= 3500:
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










