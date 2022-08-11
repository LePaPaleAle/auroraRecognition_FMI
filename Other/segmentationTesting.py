import cv2
from PIL import Image, ImageStat
from skimage.feature import canny
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

def autoCanny(grayscaleImg, sigma=0.4):
    grayscaleMedian = np.median(grayscaleImage)
    grayscaleMean = np.mean(grayscaleImage)
    lower = int(max(0, (1.0 - sigma) * grayscaleMedian))
    upper = int(min(255, (1.0 + sigma) * grayscaleMedian))
    edged = cv2.Canny(grayscaleImg, lower, upper)

    return edged


inputPath = '/home/moisio/Documents/auroraRecognition/testSetLarge/cropped/aurora/25012022_155134-0016_masked.png'
pilImage = Image.open(inputPath)
npImage = np.array(pilImage)
imageStatsRGB = ImageStat.Stat(pilImage)
imageMean = imageStatsRGB.mean
imageMedian = imageStatsRGB.median
inputImage = cv2.cvtColor(cv2.imread(inputPath), cv2.COLOR_BGR2RGB)
grayscaleImage = cv2.cvtColor(cv2.imread(inputPath), cv2.COLOR_BGR2GRAY)
blurredImageGray = cv2.GaussianBlur(grayscaleImage, (3, 3), 0)
blurredImageRGB = cv2.GaussianBlur(inputImage, (3, 3), 0)
subtractedImage = npImage[:, :, :] - imageMean
subtractedImage[subtractedImage < 0] = 0
subtractedImage = cv2.cvtColor(subtractedImage.astype(np.uint8), cv2.COLOR_BGR2RGB)
# adaptive mean thresholding
# adaptiveMean = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# adaptive gaussian mean thresholding
# adaptiveGaussian = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# automatic canny edge
# autoCannyEdges = autoCanny(grayscaleImage)

# elevationMap = sobel(grayscaleImage)
# markers = np.zeros_like(grayscaleImage)
# markers[grayscaleImage < 175] = 1
# markers[grayscaleImage >= 175] = 2
#
# segmentation = watershed(elevationMap, markers)
# segmentation = ndi.binary_fill_holes(segmentation - 1)
# segmentation = segmentation.astype(np.uint8) * 255
fig, ax = plt.subplots(2, 3)
print('Image mean:', imageStatsRGB.mean)
print('Image median:', imageStatsRGB.median)
print('Image standard deviation:', imageStatsRGB.stddev)
ax[0, 0].imshow(inputImage)
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')
ax[0, 1].imshow(grayscaleImage, cmap='gray')
ax[0, 1].set_title('Grayscale')
ax[0, 1].axis('off')
ax[0, 2].imshow(blurredImageGray, cmap='gray')
ax[0, 2].set_title('Blurred Image grayscale')
ax[0, 2].axis('off')
ax[1, 0].imshow(blurredImageRGB)
ax[1, 0].set_title('Blurred Image RGB')
ax[1, 0].axis('off')
ax[1, 1].imshow(subtractedImage)
ax[1, 1].set_title('Subtracted Image')
ax[1, 1].axis('off')
ax[1, 2].axis('off')

plt.show()

