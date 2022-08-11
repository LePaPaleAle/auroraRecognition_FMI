import shutil
import time
from PIL import Image
from numpy import array, amin, amax, full_like, zeros, where, shape, nonzero, minimum, maximum, size, invert
from glob import glob
from pytesseract.pytesseract import image_to_string
import random
from matplotlib.pyplot import subplots, subplots_adjust, legend, savefig, close
from matplotlib.lines import Line2D
from moviepy.editor import *
from subprocess import run


''' This function reads an ASC image and extracts the location, date and time from the image itself.
    The function uses prespecified (x_0, y_0, x_1, y_1) coordinates in order to crop the image from the rigt places.
    Recognizing the text is done by pytesseracts image_to_string -function '''


def getTimeStamp(image, locationCoordinates, dateCoordinates, timeCoordinates):
    # Crop the location, date and year areas from the image
    locationImage, dateImage, timeImage = image.crop(locationCoordinates).resize((220, 60)).convert('L'), \
                                          image.crop(dateCoordinates).resize((220, 60)).convert('L'),\
                                          image.crop(timeCoordinates).resize((250, 60)).convert('L')
    # Get and format the location, date and time strings
    location, date, time = ''.join(filter(str.isalnum, image_to_string(locationImage))), \
                           ''.join(filter(str.isalnum, image_to_string(dateImage))), \
                           image_to_string(timeImage).rstrip().replace('-', ':').replace('.', ':')

    # Combine the strings to the final format KEVO_YYYYMMDD_hhmmss
    yyyy_mm_dd = ''.join(['20', date[4:], date[2:4], date[:2]])
    location_yyyymmdd_hhmmss = '_'.join([location, yyyy_mm_dd, time])
    return location_yyyymmdd_hhmmss


''' This function crops the surplus area, i.e. trees and data, of the original image.
    Both the ASC image and mask Image are read as numpy arrays. The mask is a binary mask, so a
    True/False array can be created by finding the pixels that are white.
    The resulting array can then be used to select the pixels that shall be cropped. '''


def cropTrees(image, mask):
    # make numpy arrays of both the ASC image and mask
    image = array(image)
    mask = array(mask)
    # Find the pixels that are white in the mask, i.e. pixels in the ASC image that are not needed
    mask = mask == 255
    # Now, set the excessive pixels in the ASC image to black in order to crop the ASC image
    image[mask] = (0, 0, 0)
    # Get the indices where the mask is false, i.e. the area that shall be preserved
    ind = where(~mask)
    # find edges of mask to crop image (so we don't need to process all the black around the actual image)
    # amin = find minimum along an axis
    min_y, max_y, min_x, max_x = amin(ind[0]) - 1, amax(ind[0]) + 2, amin(ind[1]) - 1, amax(ind[1]) + 2
    # crop image and mask to cut away unwanted pixels
    image, mask = image[min_y:max_y, min_x:max_x, :], mask[min_y:max_y, min_x:max_x]
    return image, mask


''' This function does the calssification of the ASC image.
    R, G, B and L values are extracted and calculated from the iamge array.
    Finally, a set of conditions (classification criteria) are imposed on the image array leading to a
    True/False array for each classification class. The True/False arrays are then "stacked"
    on top of one another giving us the classification array.'''


def classifyImage(np_img_array):
    # Determine the dimensions and RGB + L values of the array
    height, width, _ = shape(np_img_array)
    classifiedImage = full_like(zeros((height, width)), False, dtype=bool)
    R, G, B = np_img_array[:, :, 0] / 255., np_img_array[:, :, 1] / 255., np_img_array[:, :, 2] / 255.
    minimum_array = minimum(minimum(R, G), B)
    maximum_array = maximum(maximum(R, G), B)
    L = (minimum_array + maximum_array) / 2

    # Calculate the classification "masks"
    green_mask = (R < G) & (R >= B) & (G > B) & (L > 0.4) & (L <= 0.98) & \
                 ((R - G) < -0.01) & ((R - B) > 0.01) & ((G - B) > 0.01)

    cyan_mask = (R < G) & (R <= B) & (G > B) & (L > 0.4) & (L <= 0.98) & \
                ((R - G) < -0.01) & ((R - B) < -0.01) & ((G - B) > 0.01)

    red_mask_1 = (R > G) & (R > B) & (G >= B) & (L >= 0.45) & (L <= 0.85) & \
                 ((R - G) > 0.01) & (R - B > 0.18) & ((G - B) > 0.018)

    red_mask_2 = (R > G) & (R > B) & (G <= B) & (L >= 0.45) & (L <= 0.85) & \
                 ((R - B) > 0.015)

    purple_mask = (R >= G) & (R <= B) & (G <= B) & (L > 0.5) & (L <= 0.98) & \
                  ((R - G) > 0.03) & ((R - B) < -0.12) & ((G - B) < -0.15)

    cloud_mask = (L < 0.85) & \
                 (((-0.01 <= (R - G)) & ((R - G) <= 0.01)) |
                 ((-0.01 <= (R - B)) & ((R - B) <= 0.01)) |
                 ((-0.01 <= (G - B)) & ((G - B) <= 0.01)))

    # Form the final classification result array that is used for determining if the picture has aurora or not
    classifiedImage[green_mask] = 1
    classifiedImage[red_mask_1] = 1
    classifiedImage[red_mask_2] = 1
    classifiedImage[purple_mask] = 1
    classifiedImage[cyan_mask] = 1
    classifiedImage[cloud_mask] = 0

    return classifiedImage


'''DEFINING GLOBAL VARIABLES'''
# (x_0, y_0, x_1, y_1) coordinates for KEVO ASC location, date and time position in image
LOCATION_COORDINATES, DATE_COORDINATES, TIME_COORDINATES = (0, 0, 52, 30), (50, 0, 134, 32), (500, 0, 608, 32)

'''THESE PATHS MUST BE CHECKED WHEN SCRIPT IS MOVED TO ANOTHER LOCATION!!'''
AURORA_PATH = '/home/users/moisio/testing/KEV/aurora'
NO_AURORA_PATH = '/home/users/moisio/testing/KEV/no-aurora'
if not os.path.exists(AURORA_PATH):
    os.mkdir(AURORA_PATH)
NO_AURORA_PATH = '/home/users/moisio/testing/MUO/no-aurora'
if not os.path.exists(NO_AURORA_PATH):
    os.mkdir(NO_AURORA_PATH)
MASK_PATH = '/home/users/moisio/auroraRecognition/mask.png'
TREE_MASK = Image.open(MASK_PATH)
AURORA, NO_AURORA = 1, 2
CAMERA_STATUS_PATH = '/home/users/moisio/testing/KEV/KEV_status.html'
VIDEO_PATHS = ['/arch/aurora/ASC/Color/KEV/movies/2022', '/arch/aurora/ASC/Color/KEV/movies/2022']
TEMP_NAME = '/home/users/moisio/testing/KEV/temp.jpg'
LATEST_IMAGE_NAME = '/home/users/moisio/testing/KEV/latest_KEV.png'

try:
    # 'Toss a coin' and choose the video folder randomly from Crystal
    coinFlip = random.randint(0, 1)
    VIDEO_PATH = VIDEO_PATHS[coinFlip]
    DATES = os.listdir(VIDEO_PATH)

    # Select the day and the video randomly as well
    DAY_INDEX = random.randint(0, len(DATES))
    DAY_VIDEO = random.choice(glob(VIDEO_PATH + '/' + DATES[DAY_INDEX] + '/*.avi'))

    # Read the random video with moviepy
    inputVideo = VideoFileClip(DAY_VIDEO)
    frameCount = int(inputVideo.duration * inputVideo.fps)
    randomFrame = random.randint(0, frameCount)
    # Extract a random frame
    inputVideo.save_frame(TEMP_NAME, t=int(randomFrame / inputVideo.fps))
    inputImage = Image.open(TEMP_NAME)

    # Get the "timestamp" of the image and rename the input ASC image according to it.
    NEW_NAME = getTimeStamp(inputImage,
                            timeCoordinates=TIME_COORDINATES,
                            dateCoordinates=DATE_COORDINATES,
                            locationCoordinates=LOCATION_COORDINATES) + '.jpg'
    os.rename(TEMP_NAME, NEW_NAME)

    # Cut the trees from the image and classify it
    img, treeMask = cropTrees(inputImage, TREE_MASK)
    R = img[:, :, 0]
    N = size(R[nonzero(R)])
    classifiedImage = classifyImage(img)

    # Calculate the amount of pixels with or without aurora
    auroraPixels = classifiedImage[invert(treeMask)]
    auroraPixelCount = auroraPixels[auroraPixels == 1].size
    noAuroraPixelCount = auroraPixels[auroraPixels == 0].size

    # Create the image that shall be displayed on the web page
    # Add the "traffic light" indicating whether the image has aurora or not with matplotlib
    handles = []
    if auroraPixelCount >= 2500:
        handles.append(Line2D([], [], marker='o', color='none', label='This image\nhas Aurora.', markersize=25,
                              markerfacecolor='g', markeredgecolor='black', markeredgewidth=3))
        handles.append(Line2D([], [], marker='o', color='none', label='This image\nhas no Aurora.', markersize=25,
                              markerfacecolor='gray', markeredgecolor='black', markeredgewidth=3))
        shutil.move(NEW_NAME, AURORA_PATH)
    else:
        handles.append(Line2D([], [], marker='o', color='none', label='This image\nhas Aurora.', markersize=25,
                              markerfacecolor='gray', markeredgecolor='black', markeredgewidth=3))
        handles.append(Line2D([], [], marker='o', color='none', label='This image\nhas no Aurora.', markersize=25,
                              markerfacecolor='red', markeredgecolor='black', markeredgewidth=3))
        shutil.move(NEW_NAME, NO_AURORA_PATH)

    newImage = Image.new(inputImage.mode, (850, inputImage.height), (255, 255, 255))
    newImage.paste(inputImage, (0, 0))
    fig, ax = subplots(1, figsize=(12, 8))
    ax.imshow(newImage)
    ax.axis('off')
    subplots_adjust(left=0.05, bottom=0, right=0.9, top=1)
    legend(handles=handles, loc=(0.75, 0.75), facecolor='white', framealpha=1, fontsize=20)
    savefig(LATEST_IMAGE_NAME)
    close()
    newImage.close()
    inputImage.close()
    endTime = time.time()
    with open(CAMERA_STATUS_PATH, 'w+') as cameraStatus:
        cameraStatus.write('Camera working.')
    run(['rsync', '--timeout=30',  '-avP', '/home/users/moisio/testing/KEV/latest_KEV.png',
        'moisio@astra.fmi.fi:/home/users/moisio/public_html/testing/KEV'])
    run(['rsync', '--timeout=30', '-avP', '/home/users/moisio/testing/KEV/KEV_status.html',
         'moisio@astra.fmi.fi:/home/users/moisio/public_html/testing/KEV'])
    # print('Classified and sent one image to Astra in {} s'.format(endTime - startTime))

except Exception as e:
    exception_type, exception_object, exception_traceback = sys.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    exception_file = open('/home/users/moisio/testing/KEV/error_log.txt', 'a+')
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    exception_file.write('\nError at {}\n'.format(current_time))
    exception_file.write('Exception type: {}\n'.format(exception_type))
    exception_file.write('File name: {}\n'.format(filename))
    exception_file.write('Line number: {}\n'.format(line_number))
    with open(CAMERA_STATUS_PATH, 'w+') as cameraStatus:
        cameraStatus.write('Camera not working!')
    run(['rsync', '--timeout=30', '-avP', '/home/users/moisio/testing/KEV/KEV_status.html',
         'moisio@astra.fmi.fi:/home/users/moisio/public_html/testing/KEV'])


