from PIL import Image
import numpy as np
from matplotlib import pyplot as plt, patches
import glob
import os


def classifyPixels(inputImage, pixelCriteria):
    handles = []
    height, width, _ = np.shape(inputImage)
    classifiedImage = np.zeros((height, width))
    cmap = plt.cm.get_cmap('Set1', len(pixelCriteria))
    colors = cmap(range(len(pixelCriteria)))[:, :3] * 255
    r_idx = inputImage[:, :, 0] / 255.
    g_idx = inputImage[:, :, 1] / 255.
    b_idx = inputImage[:, :, 2] / 255.
    N = np.size(r_idx[np.nonzero(r_idx)])
    r_min_g, r_min_b, g_min_b = r_idx - g_idx, r_idx - b_idx, g_idx - b_idx
    minimum = np.minimum(np.minimum(r_idx, g_idx), b_idx)
    maximum = np.maximum(np.maximum(r_idx, g_idx), b_idx)
    L = (minimum + maximum) / 2
    greenAuroraImage = np.array(inputImage.copy())
    green_criteria = pixelCriteria['Green Aurora']
    green_mask = (green_criteria['L_max'] >= L) & (L >= green_criteria['L_min']) & \
                   (green_criteria['R_max'] >= r_idx) & (r_idx >= green_criteria['R_min']) & \
                   (green_criteria['G_max'] >= g_idx) & (g_idx >= green_criteria['G_min']) & \
                   (green_criteria['B_max'] >= b_idx) & (b_idx >= green_criteria['B_min']) & \
                   ((green_criteria['R_B_max_1'] >= r_min_b) & (r_min_b >= green_criteria['R_B_min_1']) |
                    (green_criteria['R_B_max_2'] >= r_min_b) & (r_min_b >= green_criteria['R_B_min_2'])) & \
                   (green_criteria['R_G_max'] >= r_min_g) & (r_min_g >= green_criteria['R_G_min']) & \
                   (green_criteria['G_B_max'] >= g_min_b) & (g_min_b >= green_criteria['G_B_min']) & \
                   (g_idx > b_idx) & (g_idx > r_idx)

    classifiedImage[green_mask] = 1
    greenAuroraImage[green_mask] = colors[0]
    handles.append(patches.Patch(color=colors[0] / 255., label='Green Aurora'))

    redAuroraImage = np.array(inputImage.copy())
    red_criteria = pixelCriteria['Red Aurora']
    red_mask = (red_criteria['L_max'] >= L) & (L >= red_criteria['L_min']) & \
                   (red_criteria['R_max'] >= r_idx) & (r_idx >= red_criteria['R_min']) & \
                   (red_criteria['G_max'] >= g_idx) & (g_idx >= red_criteria['G_min']) & \
                   (red_criteria['B_max'] >= b_idx) & (b_idx >= red_criteria['B_min']) & \
                   ((red_criteria['G_B_max_1'] >= g_min_b) & (g_min_b >= red_criteria['G_B_min_1']) |
                    (red_criteria['G_B_max_2'] >= g_min_b) & (g_min_b >= red_criteria['G_B_min_2'])) & \
                   (red_criteria['R_G_max'] >= r_min_g) & (r_min_g >= red_criteria['R_G_min']) & \
                   (red_criteria['R_B_max'] >= r_min_b) & (r_min_b >= red_criteria['R_B_min']) & \
                   (r_idx > b_idx) & (r_idx > g_idx)

    classifiedImage[red_mask] = 1
    redAuroraImage[red_mask] = colors[1]
    handles.append(patches.Patch(color=colors[1] / 255., label='Red Aurora'))

    purpleAuroraImage = np.array(inputImage.copy())
    purple_criteria = pixelCriteria['Purple Aurora']
    purple_mask = (purple_criteria['L_max'] >= L) & (L >= purple_criteria['L_min']) & \
                   (purple_criteria['R_max'] >= r_idx) & (r_idx >= purple_criteria['R_min']) & \
                   (purple_criteria['G_max'] >= g_idx) & (g_idx >= purple_criteria['G_min']) & \
                   (purple_criteria['B_max'] >= b_idx) & (b_idx >= purple_criteria['B_min']) & \
                   ((purple_criteria['R_B_max_1'] >= r_min_b) & (r_min_b >= purple_criteria['R_B_min_1']) |
                    (purple_criteria['R_B_max_2'] >= r_min_b) & (r_min_b >= purple_criteria['R_B_min_2'])) & \
                   (purple_criteria['R_G_max'] >= r_min_g) & (r_min_g >= purple_criteria['R_G_min']) & \
                   (purple_criteria['G_B_max'] >= g_min_b) & (g_min_b >= purple_criteria['G_B_min']) & \
                   (r_idx > g_idx) & (b_idx > g_idx)

    classifiedImage[purple_mask] = 1
    purpleAuroraImage[purple_mask] = colors[2]
    handles.append(patches.Patch(color=colors[2] / 255., label='Purple Aurora'))

    return classifiedImage, greenAuroraImage, redAuroraImage, purpleAuroraImage, handles


criteria = {

    'Green Aurora': {'L_min': 0.325, 'L_max': 0.99,
                'R_min': 0.1, 'R_max': 0.99,
                'G_min': 0.2, 'G_max': 0.995,
                'B_min': 0.1, 'B_max': 0.99,
                'R_B_min_1': -0.99, 'R_B_max_1': -0.025,
                'R_B_min_2': 0.05, 'R_B_max_2': 0.99,
                'R_G_min': -0.99, 'R_G_max': -0.02,
                'G_B_min': 0.02, 'G_B_max': 0.99},


    'Red Aurora': {'L_min': 0.2, 'L_max': 0.95,
                           'R_min': 0.3, 'R_max': 1,
                           'G_min': 0.2, 'G_max': 0.9,
                           'B_min': 0.2, 'B_max': 0.9,
                           'R_G_min': 0.2, 'R_G_max': 0.99,
                           'R_B_min': 0.2, 'R_B_max': 0.99,
                           'G_B_min_1': 0.015, 'G_B_max_1': 0.99,
                           'G_B_min_2': -0.99, 'G_B_max_2': -0.015},

    'Purple Aurora': {'L_min': 0.2, 'L_max': 0.95,
                'R_min': 0.3, 'R_max': 1,
                'G_min': 0.2, 'G_max': 1,
                'B_min': 0.2, 'B_max': 1,
                'R_G_min': 0.03, 'R_G_max': 1,
                'R_B_min_1': -1, 'R_B_max_1': -0.05,
                'R_B_min_2': 0.2, 'R_B_max_2': 1,
                'G_B_min': -1, 'G_B_max': -0.03,
                      }

    # 'Testing': {'L_min': 0.2, 'L_max': 0.95,
    #             'R_min': 0.1, 'R_max': 0.99,
    #             'G_min': 0.2, 'G_max': 0.995,
    #             'B_min': 0.1, 'B_max': 0.99,
    #             'R_G_min': -0.99, 'R_G_max': -0.03,
    #             'R_B_min_1': -0.99, 'R_B_max_1': -0.03,
    #             'R_B_min_2': 0.06, 'R_B_max_2': 0.99,
    #             'G_B_min': 0.03, 'G_B_max': 0.99,
    #             }
}

# inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/samples3/aurora/aurora10/*.png'))
inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/*.png'))
for image in inputImages:
    ogImg = Image.open(image)
    img = np.array(ogImg)
    R = img[:, :, 0]
    N = np.size(R)
    classifiedImage, greenAuroraImage, redAuroraImage, purpleAuroraImage, handles = classifyPixels(img, pixelCriteria=criteria)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # original image
    ax[0, 0].imshow(ogImg)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('Original')
    # green
    ax[0, 1].imshow(greenAuroraImage)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('Green Aurora')
    # red
    ax[1, 0].imshow(redAuroraImage)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('Red Aurora')
    # purple image
    ax[1, 1].imshow(purpleAuroraImage)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('Purple Aurora')

    ax[0, 0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[0, 1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[1, 0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[1, 1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    auroraPixels = np.count_nonzero(classifiedImage)
    plt.suptitle(auroraPixels, color='pink')
    path = os.path.abspath(image).split('/')
    file_name = path[-1]
    path = path[:-1]
    path.append('classified_testing')
    path = '/'.join(path)
    fig.savefig('/'.join([path, file_name]), bbox_inches='tight', facecolor='black')
    # plt.show()
    plt.close()







