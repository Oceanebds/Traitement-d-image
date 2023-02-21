# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:46:37 2020

@author: capliera
"""

import skimage.io as skio
import numpy as np
from os.path import join
from skimage.transform import hough_line, hough_line_peaks
import math
from scipy import ndimage
import skimage.filters as skf
import matplotlib.pyplot as plt
import skimage.morphology as skm
from skimage.util import img_as_ubyte


plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'insurance_form.jpg'
dirpath = r'..\..\images'
filepath = join(dirpath, filename)
print(filepath)

img = skio.imread(filepath)
#img = sk.img_as_float(img)
print("Loaded image has dimensions:", img.shape)
plt.figure(), plt.subplot(2,2,1), plt.imshow(img)

# Image insurance binarization

#TO BE COMPLETED

# Hough line detection and image angle rotation computation

# TO BE COMPLETED

# Image Rotation - After image rotation, it is necessary to have a 
# binary image with values 0 or 255

# TO BE COMPLETED

# Image line restoration with morphologic operators
# Provide in the variable alligned_img a binary image with 255 for the
# background pixels and 0 for the Table information and lines

# Erode with vertical Structure Elt, then close with vertical SE
SE = np.ones((35,1),np.uint8)
imgRotNegErode = skm.erosion(255-aligned_img, SE)
SE = np.ones((201,1),np.uint8)
imgRotNegErodeClose = skm.closing(imgRotNegErode,SE)
for i in range(aligned_img.shape[0]):
    for j in range(aligned_img.shape[1]):
        if (imgRotNegErodeClose[i,j]>255-aligned_img[i,j]):
            aligned_img[i,j] = imgRotNegErodeClose[i,j]

# Erode with horizontal SE, then close with horizontal SE
SE = np.ones((1,49),np.uint8)
imgRotNegErode = skm.erosion(255-aligned_img, SE)
SE = np.ones((1,301),np.uint8)
imgRotNegErodeClose = skm.closing(imgRotNegErode,SE)
for i in range(aligned_img.shape[0]):
    for j in range(aligned_img.shape[1]):
        if (imgRotNegErodeClose[i,j]>255-aligned_img[i,j]):
            aligned_img[i,j] = imgRotNegErodeClose[i,j]
plt.figure(), ptl.imshow(img)
plt.figure(), plt.imshow(aligned_img)