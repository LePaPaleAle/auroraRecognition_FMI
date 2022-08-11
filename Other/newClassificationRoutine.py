from PIL import Image
import numpy as np
from matplotlib import pyplot as plt, patches
import glob
import os


def makeAuroraMask(noAuroraMask, height, width, categories, colors, coloredImage, handles, L, R, G, B,
                     R_min_G, R_min_B, G_min_B):

    auroraMask = np.full_like(np.zeros((height, width)), False, dtype=bool)
    green_criteria = categories['Green Aurora']
    green_mask = (green_criteria['L_max'] >= L) & (L >= green_criteria['L_min']) & \
                 (green_criteria['R_max'] >= R) & (R >= green_criteria['R_min']) & \
                 (green_criteria['G_max'] >= G) & (G >= green_criteria['G_min']) & \
                 (green_criteria['B_max'] >= B) & (B >= green_criteria['B_min']) & \
                 ((green_criteria['R_B_max_1'] >= R_min_B) & (R_min_B >= green_criteria['R_B_min_1']) |
                  (green_criteria['R_B_max_2'] >= R_min_B) & (R_min_B >= green_criteria['R_B_min_2'])) & \
                 (green_criteria['R_G_max'] >= R_min_G) & (R_min_G >= green_criteria['R_G_min']) & \
                 (green_criteria['G_B_max'] >= G_min_B) & (G_min_B >= green_criteria['G_B_min']) & \
                 (G > B) & (G > R)

    auroraMask[green_mask] = 1
    coloredImage[green_mask] = colors[0]
    handles.append(patches.Patch(color=colors[0] / 255., label='Green Aurora'))

    red_criteria = categories['Red Aurora']
    red_mask = (red_criteria['L_max'] >= L) & (L >= red_criteria['L_min']) & \
               (red_criteria['R_max'] >= R) & (R >= red_criteria['R_min']) & \
               (red_criteria['G_max'] >= G) & (G >= red_criteria['G_min']) & \
               (red_criteria['B_max'] >= B) & (B >= red_criteria['B_min']) & \
               ((red_criteria['G_B_max_1'] >= G_min_B) & (G_min_B >= red_criteria['G_B_min_1']) |
                (red_criteria['G_B_max_2'] >= G_min_B) & (G_min_B >= red_criteria['G_B_min_2'])) & \
               (red_criteria['R_G_max'] >= R_min_G) & (R_min_G >= red_criteria['R_G_min']) & \
               (red_criteria['R_B_max'] >= R_min_B) & (R_min_B >= red_criteria['R_B_min']) & \
               (R > B) & (R > G)

    auroraMask[red_mask] = 1
    coloredImage[red_mask] = colors[1]
    handles.append(patches.Patch(color=colors[1] / 255., label='Red Aurora'))

    purple_criteria = categories['Purple Aurora']
    purple_mask = (purple_criteria['L_max'] >= L) & (L >= purple_criteria['L_min']) & \
                  (purple_criteria['R_max'] >= R) & (R >= purple_criteria['R_min']) & \
                  (purple_criteria['G_max'] >= G) & (G >= purple_criteria['G_min']) & \
                  (purple_criteria['B_max'] >= B) & (B >= purple_criteria['B_min']) & \
                  ((purple_criteria['R_B_max_1'] >= R_min_B) & (R_min_B >= purple_criteria['R_B_min_1']) |
                   (purple_criteria['R_B_max_2'] >= R_min_B) & (R_min_B >= purple_criteria['R_B_min_2'])) & \
                  (purple_criteria['R_G_max'] >= R_min_G) & (R_min_G >= purple_criteria['R_G_min']) & \
                  (purple_criteria['G_B_max'] >= G_min_B) & (G_min_B >= purple_criteria['G_B_min']) & \
                  (R > G) & (B > G)

    auroraMask[purple_mask] = 1
    coloredImage[purple_mask] = colors[2]
    handles.append(patches.Patch(color=colors[2] / 255., label='Purple Aurora'))

    auroraMask[noAuroraMask] = 0

    return auroraMask


def makeNoAuroraMask(height, width, categories, colors, coloredImage, handles, L, R, G, B,
                     R_min_G, R_min_B, G_min_B):

    baseMask = np.full_like(np.zeros((height, width)), False, dtype=bool)

    no_aurora_1_criteria = categories['No Aurora 1']
    no_aurora_1_mask = (no_aurora_1_criteria['L_max'] >= L) & (L >= no_aurora_1_criteria['L_min']) & \
                       (no_aurora_1_criteria['R_max'] >= R) & (R >= no_aurora_1_criteria['R_min']) & \
                       (no_aurora_1_criteria['G_max'] >= G) & (G >= no_aurora_1_criteria['G_min']) & \
                       (no_aurora_1_criteria['B_max'] >= B) & (B >= no_aurora_1_criteria['B_min']) & \
                       (no_aurora_1_criteria['R_G_max'] >= R_min_G) & (R_min_G >= no_aurora_1_criteria['R_G_min']) & \
                       (no_aurora_1_criteria['R_B_max'] >= R_min_B) & (R_min_B >= no_aurora_1_criteria['R_B_min']) & \
                       (no_aurora_1_criteria['G_B_max'] >= G_min_B) & (G_min_B >= no_aurora_1_criteria['G_B_min'])

    baseMask[no_aurora_1_mask] = 1
    coloredImage[no_aurora_1_mask] = colors[3]
    handles.append(patches.Patch(color=colors[3] / 255., label='No Aurora 1'))

    no_aurora_2_criteria = categories['No Aurora 2']
    no_aurora_2_mask = (no_aurora_2_criteria['L_max'] >= L) & (L >= no_aurora_2_criteria['L_min']) & \
                    (no_aurora_2_criteria['R_max'] >= R) & (R >= no_aurora_2_criteria['R_min']) & \
                    (no_aurora_2_criteria['G_max'] >= G) & (G >= no_aurora_2_criteria['G_min']) & \
                    (no_aurora_2_criteria['B_max'] >= B) & (B >= no_aurora_2_criteria['B_min']) & \
                    (no_aurora_2_criteria['R_G_max'] >= R_min_G) & (R_min_G >= no_aurora_2_criteria['R_G_min']) & \
                    (no_aurora_2_criteria['R_B_max'] >= R_min_B) & (R_min_B >= no_aurora_2_criteria['R_B_min']) & \
                    (no_aurora_2_criteria['G_B_max'] >= G_min_B) & (G_min_B >= no_aurora_2_criteria['G_B_min'])

    coloredImage[no_aurora_2_mask] = colors[4]
    handles.append(patches.Patch(color=colors[4] / 255., label='No Aurora 2'))
    baseMask[no_aurora_2_mask] = 1

    return baseMask


def classifyPixels(inputImage, pixelCriteria):
    handles = []
    height, width, _ = np.shape(inputImage)
    classifiedImage = np.zeros((height, width))
    coloredImage = np.array(inputImage.copy())
    cmap = plt.cm.get_cmap('RdYlGn', len(pixelCriteria))
    colors = cmap(range(len(pixelCriteria)))[:, :3] * 255
    r_idx = inputImage[:, :, 0] / 255.
    g_idx = inputImage[:, :, 1] / 255.
    b_idx = inputImage[:, :, 2] / 255.
    r_min_g, r_min_b, g_min_b = r_idx - g_idx, r_idx - b_idx, g_idx - b_idx
    # r_div_g, r_div_b, g_div_b = np.divide(r_idx, g_idx, out=np.zeros(r_idx.shape, dtype=float), where=g_idx != 0),\
    #                             np.divide(r_idx, b_idx, out=np.zeros(r_idx.shape, dtype=float), where=b_idx != 0),\
    #                             np.divide(g_idx, b_idx, out=np.zeros(g_idx.shape, dtype=float), where=b_idx != 0)
    minimum = np.minimum(np.minimum(r_idx, g_idx), b_idx)
    maximum = np.maximum(np.maximum(r_idx, g_idx), b_idx)
    L = (minimum + maximum) / 2
    noAuroraMask = makeNoAuroraMask(height, width, pixelCriteria, colors, coloredImage, handles, L, r_idx, g_idx, b_idx,
                                    r_min_g, r_min_b, g_min_b)
    auroraMask = makeAuroraMask(noAuroraMask, height, width, pixelCriteria, colors, coloredImage, handles, L,
                                r_idx, g_idx, b_idx, r_min_g, r_min_b, g_min_b)

    classifiedImage[noAuroraMask] = 2
    classifiedImage[auroraMask] = 1

    return classifiedImage, coloredImage, handles


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

    # crop image and mask to cut away unwanted pixels
    image = image[min_y:max_y, min_x:max_x, :]
    mask = mask[min_y:max_y, min_x:max_x]

    return image, mask


criteria = {
    'Green Aurora': {'L_min': 0.32, 'L_max': 0.99,
                     'R_min': 0.1, 'R_max': 0.99,
                     'G_min': 0.2, 'G_max': 0.995,
                     'B_min': 0.1, 'B_max': 0.99,
                     'R_B_min_1': -0.99, 'R_B_max_1': -0.02,
                     'R_B_min_2': 0.05, 'R_B_max_2': 0.99,
                     'R_G_min': -0.99, 'R_G_max': -0.02,
                     'G_B_min': 0.02, 'G_B_max': 0.99},

    'Red Aurora': {'L_min': 0.2, 'L_max': 0.95,
                   'R_min': 0.3, 'R_max': 1,
                   'G_min': 0.2, 'G_max': 0.9,
                   'B_min': 0.2, 'B_max': 0.9,
                   'R_G_min': 0.2, 'R_G_max': 0.99,
                   'R_B_min': 0.2, 'R_B_max': 0.99,
                   'G_B_min_1': 0.012, 'G_B_max_1': 0.99,
                   'G_B_min_2': -0.99, 'G_B_max_2': -0.012},

    'Purple Aurora': {'L_min': 0.2, 'L_max': 0.95,
                      'R_min': 0.3, 'R_max': 1,
                      'G_min': 0.2, 'G_max': 1,
                      'B_min': 0.2, 'B_max': 1,
                      'R_G_min': 0.02, 'R_G_max': 1,
                      'R_B_min_1': -1, 'R_B_max_1': -0.05,
                      'R_B_min_2': 0.2, 'R_B_max_2': 1,
                      'G_B_min': -1, 'G_B_max': -0.02,
                      },

    'No Aurora 1': {'L_min': 0.1, 'L_max': 1,
                  'R_min': 0.1, 'R_max': 1,
                  'G_min': 0.1, 'G_max': 1,
                  'B_min': 0.1, 'B_max': 1,
                  'R_G_min': 0.0, 'R_G_max': 0.4,
                  'R_B_min': 0.0, 'R_B_max': 0.4,
                  'G_B_min': 0.0, 'G_B_max': 0.4,
                  },

    'No Aurora 2': {'L_min': 0.1, 'L_max': 1,
                  'R_min': 0.1, 'R_max': 1,
                  'G_min': 0.1, 'G_max': 1,
                  'B_min': 0.1, 'B_max': 1,
                  'R_G_min': -0.1, 'R_G_max': 0.0,
                  'R_B_min': -0.15, 'R_B_max': 0.0,
                  'G_B_min': -0.1, 'G_B_max': 0.0,
                  },

}

inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/*.png'))
# inputImages = ['/home/moisio/Documents/auroraRecognition/testSetLarge/01022022_181644-0021.png']
treeMask = Image.open('/home/moisio/Documents/auroraRecognition/mask.png')
for image in inputImages:
    img = Image.open(image)
    img, mask = cropTrees(img, treeMask)
    R = img[:, :, 0]
    N = np.size(R[np.nonzero(R)])
    classifiedImage, coloredImage, handles = classifyPixels(img, pixelCriteria=criteria)
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
    path.append('classified_01072022_modified')
    # check which pixels are categorised as moon,clear,aurora
    # total number of pixels in image
    auroraPixels = classifiedImage[np.invert(mask)]
    # auroraPixelCount = classifiedImage[np.nonzero(classifiedImage)].size
    auroraPixelCount = auroraPixels[auroraPixels == 1].size
    noAuroraPixelCount = auroraPixels[auroraPixels == 2].size
    # noAuroraPixelCount = classifiedImage[classifiedImage == 0].size
    # if certain number of pixels are in certain category, image is categorized as this category
    if auroraPixelCount >= 1000:
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







