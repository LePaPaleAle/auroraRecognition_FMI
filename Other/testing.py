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
    testAuroraImage1 = np.array(inputImage.copy())
    test_criteria1 = pixelCriteria['Testing 1']
    test_mask1 = (test_criteria1['L_max'] >= L) & (L >= test_criteria1['L_min']) & \
                   (test_criteria1['R_max'] >= r_idx) & (r_idx >= test_criteria1['R_min']) & \
                   (test_criteria1['G_max'] >= g_idx) & (g_idx >= test_criteria1['G_min']) & \
                   (test_criteria1['B_max'] >= b_idx) & (b_idx >= test_criteria1['B_min']) & \
                   (test_criteria1['R_B_max'] >= r_min_b) & (r_min_b >= test_criteria1['R_B_min']) & \
                   (test_criteria1['R_G_max'] >= r_min_g) & (r_min_g >= test_criteria1['R_G_min']) & \
                   (test_criteria1['G_B_max'] >= g_min_b) & (g_min_b >= test_criteria1['G_B_min'])

    classifiedImage[test_mask1] = 1
    testAuroraImage1[test_mask1] = colors[0]
    handles.append(patches.Patch(color=colors[0] / 255., label='Test'))

    return classifiedImage, testAuroraImage1, handles


criteria = {

    # 'Green Aurora': {'L_min': 0.325, 'L_max': 0.99,
    #             'R_min': 0.1, 'R_max': 0.99,
    #             'G_min': 0.2, 'G_max': 0.995,
    #             'B_min': 0.1, 'B_max': 0.99,
    #             'R_B_min_1': -0.99, 'R_B_max_1': -0.025,
    #             'R_B_min_2': 0.05, 'R_B_max_2': 0.99,
    #             'R_G_min': -0.99, 'R_G_max': -0.02,
    #             'G_B_min': 0.02, 'G_B_max': 0.99},
    #
    #
    # 'Red Aurora': {'L_min': 0.2, 'L_max': 0.95,
    #                        'R_min': 0.3, 'R_max': 1,
    #                        'G_min': 0.2, 'G_max': 0.9,
    #                        'B_min': 0.2, 'B_max': 0.9,
    #                        'R_G_min': 0.2, 'R_G_max': 0.99,
    #                        'R_B_min': 0.2, 'R_B_max': 0.99,
    #                        'G_B_min_1': 0.015, 'G_B_max_1': 0.99,
    #                        'G_B_min_2': -0.99, 'G_B_max_2': -0.015},
    #
    # 'Purple Aurora': {'L_min': 0.2, 'L_max': 0.95,
    #             'R_min': 0.3, 'R_max': 1,
    #             'G_min': 0.2, 'G_max': 1,
    #             'B_min': 0.2, 'B_max': 1,
    #             'R_G_min': 0.03, 'R_G_max': 1,
    #             'R_B_min_1': -1, 'R_B_max_1': -0.05,
    #             'R_B_min_2': 0.2, 'R_B_max_2': 1,
    #             'G_B_min': -1, 'G_B_max': -0.03,
    #                   }

    'Testing 2': {'L_min': 0.1, 'L_max': 1,
                'R_min': 0.1, 'R_max': 1,
                'G_min': 0.1, 'G_max': 1,
                'B_min': 0.1, 'B_max': 1,
                'R_G_min': 0.0, 'R_G_max': 0.1,
                'R_B_min': 0.0, 'R_B_max': 0.1,
                'G_B_min': 0.0, 'G_B_max': 0.1,
                },

    # 'Testing 1': {'L_min': 0.1, 'L_max': 1,
    #               'R_min': 0.1, 'R_max': 1,
    #               'G_min': 0.1, 'G_max': 1,
    #               'B_min': 0.1, 'B_max': 1,
    #               'R_G_min': -0.2, 'R_G_max': -0.0,
    #               'R_B_min': -0.25, 'R_B_max': -0.0,
    #               'G_B_min': -0.2, 'G_B_max': -0.0,
    #               },
}

# inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/samples3/aurora/aurora10/*.png'))
inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/testSetLarge/*.png'))
for image in inputImages:
    ogImg = Image.open(image)
    img = np.array(ogImg)
    R = img[:, :, 0]
    N = np.size(R)
    classifiedImage, testAuroraImage1, handles = classifyPixels(img, pixelCriteria=criteria)
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    # original image
    ax[0].imshow(ogImg)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('1')
    # green
    ax[1].imshow(testAuroraImage1)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('2')

    ax[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
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







