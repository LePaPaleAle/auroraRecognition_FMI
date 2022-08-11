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
    r_div_g, r_div_b, g_div_b = r_idx / g_idx, r_idx / b_idx, g_idx / b_idx
    minimum = np.minimum(np.minimum(r_idx, g_idx), b_idx)
    maximum = np.maximum(np.maximum(r_idx, g_idx), b_idx)
    L = (minimum + maximum) / 2
    diffusedAuroraImage = np.array(inputImage.copy())
    diffused_criteria = pixelCriteria['Diffused Aurora']
    diffused_mask = (diffused_criteria['L_max'] >= L) & (L >= diffused_criteria['L_min']) & \
                    (diffused_criteria['R_max'] >= r_idx) & (r_idx >= diffused_criteria['R_min']) & \
                    (diffused_criteria['G_max'] >= g_idx) & (g_idx >= diffused_criteria['G_min']) & \
                    (diffused_criteria['B_max'] >= b_idx) & (b_idx >= diffused_criteria['B_min'])

    classifiedImage[diffused_mask] = 1
    diffusedAuroraImage[diffused_mask] = colors[0]
    handles.append(patches.Patch(color=colors[0] / 255., label='Diffused Aurora'))

    greenAurora1Image = np.array(inputImage.copy())
    green_aurora_1_criteria = pixelCriteria['Green Aurora 1']
    green_mask_1 = (green_aurora_1_criteria['L_max'] >= L) & (L >= green_aurora_1_criteria['L_min']) & \
                   (green_aurora_1_criteria['R_max'] >= r_idx) & (r_idx >= green_aurora_1_criteria['R_min']) & \
                   (green_aurora_1_criteria['G_max'] >= g_idx) & (g_idx >= green_aurora_1_criteria['G_min']) & \
                   (green_aurora_1_criteria['B_max'] >= b_idx) & (b_idx >= green_aurora_1_criteria['B_min']) & \
                   (g_idx > r_idx) & (r_idx > b_idx)

    classifiedImage[green_mask_1] = 2
    greenAurora1Image[green_mask_1] = colors[1]
    handles.append(patches.Patch(color=colors[1] / 255., label='Green Aurora 1'))

    greenAurora2Image = np.array(inputImage.copy())
    green_aurora_2_criteria = pixelCriteria['Green Aurora 2']
    green_mask_2 = (green_aurora_2_criteria['L_max'] >= L) & (L >= green_aurora_2_criteria['L_min']) & \
                   (green_aurora_2_criteria['R_max'] >= r_idx) & (r_idx >= green_aurora_2_criteria['R_min']) & \
                   (green_aurora_2_criteria['G_max'] >= g_idx) & (g_idx >= green_aurora_2_criteria['G_min']) & \
                   (green_aurora_2_criteria['B_max'] >= b_idx) & (b_idx >= green_aurora_2_criteria['B_min']) & \
                   (g_idx > b_idx) & (b_idx > r_idx)

    classifiedImage[green_mask_2] = 5
    greenAurora2Image[green_mask_2] = colors[4]
    handles.append(patches.Patch(color=colors[4] / 255., label='Green Aurora 2'))

    red_criteria = pixelCriteria['Red Aurora']
    red_mask = (red_criteria['L_max'] >= L) & (L >= red_criteria['L_min']) & \
               (red_criteria['R_max'] >= r_idx) & (r_idx >= red_criteria['R_min']) & \
               (red_criteria['G_max'] >= g_idx) & (g_idx >= red_criteria['G_min']) & \
               (red_criteria['B_max'] >= b_idx) & (b_idx >= red_criteria['B_min']) & \
               (red_criteria['R_min_G_max'] >= r_min_g) & (r_min_g >= red_criteria['R_min_G_min']) & \
               (red_criteria['R_min_B_max'] >= r_min_b) & (r_min_b >= red_criteria['R_min_B_min']) & \
               (red_criteria['G_min_B_max'] >= g_min_b) & (g_min_b >= red_criteria['G_min_B_min']) & \
               (r_idx > g_idx) & (r_idx > b_idx)

    redAuroraImage = np.array(inputImage.copy())
    if np.size(red_mask[red_mask == True]) < 0.25 * N:
        classifiedImage[red_mask] = 3
        redAuroraImage[red_mask] = colors[2]
    handles.append(patches.Patch(color=colors[2] / 255., label='Red Aurora'))

    purpleAuroraImage = np.array(inputImage.copy())
    purple_criteria = pixelCriteria['Purple Aurora']
    purple_mask = (purple_criteria['L_max'] >= L) & (L >= purple_criteria['L_min']) & \
                  (purple_criteria['R_max'] >= r_idx) & (r_idx >= purple_criteria['R_min']) & \
                  (purple_criteria['G_max'] >= g_idx) & (g_idx >= purple_criteria['G_min']) & \
                  (purple_criteria['B_max'] >= b_idx) & (b_idx >= purple_criteria['B_min']) & \
                  (purple_criteria['R_min_G_max'] >= r_min_g) & (r_min_g >= purple_criteria['R_min_G_min']) & \
                  (purple_criteria['R_min_B_max'] >= r_min_b) & (r_min_b >= purple_criteria['R_min_B_min']) & \
                  (purple_criteria['G_min_B_max'] >= g_min_b) & (g_min_b >= purple_criteria['G_min_B_min']) & \
                  (b_idx > r_idx) & (r_idx > g_idx)

    classifiedImage[purple_mask] = 4
    purpleAuroraImage[purple_mask] = colors[3]
    handles.append(patches.Patch(color=colors[3] / 255., label='Purple Aurora'))

    clouds1Image = np.array(inputImage.copy())
    clouds_1_criteria = pixelCriteria['Clouds 1']
    clouds_1_mask = (clouds_1_criteria['L_max'] >= L) & (L >= clouds_1_criteria['L_min']) & \
                    (clouds_1_criteria['R_max'] >= r_idx) & (r_idx >= clouds_1_criteria['R_min']) & \
                    (clouds_1_criteria['G_max'] >= g_idx) & (g_idx >= clouds_1_criteria['G_min']) & \
                    (clouds_1_criteria['B_max'] >= b_idx) & (b_idx >= clouds_1_criteria['B_min']) & \
                    (clouds_1_criteria['R_min_G_max'] >= r_min_g) & (r_min_g >= clouds_1_criteria['R_min_G_min']) & \
                    (clouds_1_criteria['R_min_B_max'] >= r_min_b) & (r_min_b >= clouds_1_criteria['R_min_B_min']) & \
                    (clouds_1_criteria['G_min_B_max'] >= g_min_b) & (g_min_b >= clouds_1_criteria['G_min_B_min']) & \
                    (g_idx > b_idx)

    classifiedImage[clouds_1_mask] = 6
    clouds1Image[clouds_1_mask] = colors[5]
    handles.append(patches.Patch(color=colors[5] / 255., label='Clouds 1'))

    clouds2Image = np.array(inputImage.copy())
    clouds_2_criteria = pixelCriteria['Clouds 2']
    clouds_2_mask = (clouds_2_criteria['L_max'] >= L) & (L >= clouds_2_criteria['L_min']) & \
                    (clouds_2_criteria['R_max'] >= r_idx) & (r_idx >= clouds_2_criteria['R_min']) & \
                    (clouds_2_criteria['G_max'] >= g_idx) & (g_idx >= clouds_2_criteria['G_min']) & \
                    (clouds_2_criteria['B_max'] >= b_idx) & (b_idx >= clouds_2_criteria['B_min']) & \
                    (clouds_2_criteria['R_min_G_max'] >= r_min_g) & (r_min_g >= clouds_2_criteria['R_min_G_min']) & \
                    (clouds_2_criteria['R_min_B_max'] >= r_min_b) & (r_min_b >= clouds_2_criteria['R_min_B_min']) & \
                    (clouds_2_criteria['G_min_B_max'] >= g_min_b) & (g_min_b >= clouds_2_criteria['G_min_B_min']) & \
                    (b_idx > g_idx)

    classifiedImage[clouds_2_mask] = 7
    clouds2Image[clouds_2_mask] = colors[6]
    handles.append(patches.Patch(color=colors[6] / 255., label='Clouds 2'))

    moonImage = np.array(inputImage.copy())
    moon_criteria = pixelCriteria['Moon']
    moon_mask = (moon_criteria['L_max'] >= L) & (L >= moon_criteria['L_min']) & \
                   (moon_criteria['R_max'] >= r_idx) & (r_idx >= moon_criteria['R_min']) & \
                   (moon_criteria['G_max'] >= g_idx) & (g_idx >= moon_criteria['G_min']) & \
                   (moon_criteria['B_max'] >= b_idx) & (b_idx >= moon_criteria['B_min']) & \
                   (moon_criteria['R_div_G_max'] >= r_div_g) & (r_div_g >= moon_criteria['R_div_G_min']) & \
                   (moon_criteria['R_div_B_max'] >= r_div_b) & (r_div_b >= moon_criteria['R_div_B_min']) & \
                   (moon_criteria['G_div_B_max'] >= g_div_b) & (g_div_b >= moon_criteria['G_div_B_min'])

    classifiedImage[moon_mask] = 8
    moonImage[moon_mask] = colors[7]
    handles.append(patches.Patch(color=colors[7] / 255., label='Moon'))
    
    darkSkyImage = np.array(inputImage.copy())
    dark_sky_criteria = pixelCriteria['Dark Sky']
    dark_sky_mask = (dark_sky_criteria['L_max'] >= L) & (L >= dark_sky_criteria['L_min']) & \
                (dark_sky_criteria['R_max'] >= r_idx) & (r_idx >= dark_sky_criteria['R_min']) & \
                (dark_sky_criteria['G_max'] >= g_idx) & (g_idx >= dark_sky_criteria['G_min']) & \
                (dark_sky_criteria['B_max'] >= b_idx) & (b_idx >= dark_sky_criteria['B_min']) & \
                (dark_sky_criteria['R_min_G_max'] >= r_min_g) & (r_min_g >= dark_sky_criteria['R_min_G_min']) & \
                (dark_sky_criteria['R_min_B_max'] >= r_min_b) & (r_min_b >= dark_sky_criteria['R_min_B_min']) & \
                (dark_sky_criteria['G_min_B_max'] >= g_min_b) & (g_min_b >= dark_sky_criteria['G_min_B_min']) & \
                (b_idx > g_idx)
    
    classifiedImage[dark_sky_mask] = 9
    darkSkyImage[dark_sky_mask] = colors[8]
    handles.append(patches.Patch(color=colors[8] / 255., label='Dark Sky'))

    return classifiedImage, diffusedAuroraImage, greenAurora1Image, greenAurora2Image, redAuroraImage, \
           purpleAuroraImage, clouds1Image, clouds2Image, moonImage, darkSkyImage, handles


criteria = {
            'Diffused Aurora': {'L_min': 0.55, 'L_max': 0.65,
                                'R_min': 0.55, 'R_max': 0.65,
                                'G_min': 0.55, 'G_max': 0.65,
                                'B_min': 0.55, 'B_max': 0.65,
                                'R_G_min': 0.85, 'R_G_max': 1,
                                'R_B_min': 0.85, 'R_B_max': 1.1,
                                'G_B_min': 0.85, 'G_B_max': 1.25},

            'Green Aurora 1': {'L_min': 0.35, 'L_max': 0.99,
                               'R_min': 0.2, 'R_max': 0.99,
                               'G_min': 0.25, 'G_max': 0.99,
                               'B_min': 0.25, 'B_max': 0.99,
                               'R_G_min': 0.6, 'R_G_max': 0.7,
                               'R_B_min': 0.7, 'R_B_max': 0.8,
                               'G_B_min': 1.1, 'G_B_max': 1.2},

            'Green Aurora 2': {'L_min': 0.4, 'L_max': 0.98,
                               'R_min': 0.5, 'R_max': 0.99,
                               'G_min': 0.5, 'G_max': 0.995,
                               'B_min': 0.5, 'B_max': 0.99,
                               'R_min_G_min': -0.5, 'R_min_G_max': 0.2,
                               'R_min_B_min': -0.5, 'R_min_B_max': 0.2,
                               'G_min_B_min': -0.5, 'G_min_B_max': 0.2},

            'Red Aurora': {'L_min': 0.3, 'L_max': 0.7,
                           'R_min': 0.3, 'R_max': 1,
                           'G_min': 0.28, 'G_max': 0.6,
                           'B_min': 0.28, 'B_max': 0.6,
                           'R_min_G_min': 0, 'R_min_G_max': 0.4,
                           'R_min_B_min': 0, 'R_min_B_max': 0.4,
                           'G_min_B_min': 0, 'G_min_B_max': 0.2},

            'Purple Aurora': {'L_min': 0.5, 'L_max': 0.9,
                              'R_min': 0.4, 'R_max': 0.95,
                              'G_min': 0.4, 'G_max': 0.65,
                              'B_min': 0.4, 'B_max': 0.95,
                              'R_min_G_min': -0.5, 'R_min_G_max': 0.2,
                              'R_min_B_min': -0.5, 'R_min_B_max': 0.2,
                              'G_min_B_min': -0.5, 'G_min_B_max': 0.2},

            'Clouds 1': {'L_min': 0.05, 'L_max': 0.95,
                         'R_min': 0.5, 'R_max': 0.9,
                         'G_min': 0.5, 'G_max': 0.7,
                         'B_min': 0.5, 'B_max': 0.9,
                         'R_min_G_min': -0.5, 'R_min_G_max': 0.5,
                         'R_min_B_min': -0.5, 'R_min_B_max': 0.5,
                         'G_min_B_min': -0.5, 'G_min_B_max': 0.5},

            'Clouds 2': {'L_min': 0.05, 'L_max': 0.98,
                         'R_min': 0.05, 'R_max': 0.7,
                         'G_min': 0.05, 'G_max': 0.9,
                         'B_min': 0.05, 'B_max': 0.9,
                         'R_min_G_min': -0.7, 'R_min_G_max': 1.7,
                         'R_min_B_min': -0.7, 'R_min_B_max': 1.7,
                         'G_min_B_min': -0.7, 'G_min_B_max': 1.7},

            'Moon': {'L_min': 0.8, 'L_max': 1,
                     'R_min': 0.8, 'R_max': 1,
                     'G_min': 0.8, 'G_max': 1,
                     'B_min': 0.8, 'B_max': 1,
                     'R_div_G_min': 0.9, 'R_div_G_max': 1.1,
                     'R_div_B_min': 0.9, 'R_div_B_max': 1.1,
                     'G_div_B_min': 0.9, 'G_div_B_max': 1.1},

            'Dark Sky': {'L_min': 0.01, 'L_max': 0.7,
                         'R_min': 0.01, 'R_max': 0.6,
                         'G_min': 0.01, 'G_max': 0.6,
                         'B_min': 0.01, 'B_max': 0.6,
                         'R_min_G_min': -0.6, 'R_min_G_max': 0.5,
                         'R_min_B_min': -0.6, 'R_min_B_max': 0.5,
                         'G_min_B_min': -0.6, 'G_min_B_max': 0.5},

}

# inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/samples3/aurora/aurora10/*.png'))
inputImages = glob.glob('/home/moisio/Documents/auroraRecognition/testSetSmall/*.png')
# inputImages = sorted(glob.glob('/home/moisio/Documents/auroraRecognition/samples3/no-aurora/tough/*.png'))
# inputImages = ['/home/moisio/Documents/auroraRecognition/samples3/hard/12022022_225837-0003 (copy).png',
#                '/home/moisio/Documents/auroraRecognition/samples3/hard/12022022_225837-0011.png',
#                '/home/moisio/Documents/auroraRecognition/samples3/hard/15022022_180658-0007.png',
#                '/home/moisio/Documents/auroraRecognition/samples3/hard/19122021_014005-0016.png',
#                '/home/moisio/Documents/auroraRecognition/samples3/hard/22102021_235359-0008.png']
for image in inputImages:
    img = np.array(Image.open(image))
    R = img[:, :, 0]
    N = np.size(R)
    classifiedImage, diffusedImage, green1Image, green2Image, redImage, purpleImage, clouds1Image, clouds2Image, \
    moonImage, darkSkyImage, handles = classifyPixels(img, pixelCriteria=criteria)
    diffusedImage = Image.fromarray(diffusedImage)
    green1Image = Image.fromarray(green1Image)
    green2Image = Image.fromarray(green2Image)
    redImage = Image.fromarray(redImage)
    purpleImage = Image.fromarray(purpleImage)
    clouds1Image = Image.fromarray(clouds1Image)
    clouds2Image = Image.fromarray(clouds2Image)
    moonImage = Image.fromarray(moonImage)
    darkSkyImage = Image.fromarray(darkSkyImage)
    fig, ax = plt.subplots(3, 3, figsize=(12, 8))
    # original image
    ax[0, 0].imshow(diffusedImage)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    # masked image
    ax[0, 1].imshow(green1Image)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[0, 2].imshow(green2Image)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[1, 0].imshow(redImage)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[1, 1].imshow(purpleImage)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[1, 2].imshow(clouds1Image)
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[2, 0].imshow(clouds2Image)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[2, 1].imshow(moonImage)
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    ax[2, 2].imshow(darkSkyImage)
    ax[2, 2].set_xticks([])
    ax[2, 2].set_yticks([])
    ax[2, 2].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    # plt.subplots_adjust(wspace=0.1, hspace=0.5)
    # plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)
    # check which pixels are categorised as moon,clear,aurora
    # total number of pixels in image
    # if folder for category doesn't exist, create folder
    plt.show()







