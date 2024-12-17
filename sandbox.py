import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


def primitive_thresholding(image, lower, upper):
    '''Thresholding using a simple lower and upper threshold'''
    _, thresh_mask = cv.threshold(image, lower, upper, cv.THRESH_BINARY_INV)
    return thresh_mask

def adaptive_thresholding(image, droplet_size, constant):
    '''Adaptive thresholding using cv2.adaptiveThreshold'''
    thresh_mask = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, droplet_size, constant)
    return thresh_mask



# read the image
img = cv.imread('Rain Datasets/flash downsized/DSC_0565.png')

# image to grayscale
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply primitive thresholding
primitive_mask = primitive_thresholding(img, 250, 255)
# apply adaptive thresholding
mask = adaptive_thresholding(img, 11, 2)

# display img, primitive_mask, mask side by side
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Image grayscaled')
plt.subplot(132)
plt.imshow(primitive_mask, cmap='gray')
plt.title('Primitive Thresholding')
plt.subplot(133)
plt.imshow(mask, cmap='gray')
plt.title('Adaptive Thresholding')
plt.show()