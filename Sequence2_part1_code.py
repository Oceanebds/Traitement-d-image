# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:24:57 2020

@author: capliera
"""

import math
import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.exposure as ske
from os.path import join


# All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and
# the image figure size

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6, 4)

# Image loadind and display -Histogram computation
# Plot in the same figure the image and its histogram

filename = r'Hawkes_Bay.jpg'
dirpath = r'..\images'
filepath = join(dirpath, filename)
print(filepath)

# TO BE COMPLETED

# Histogram stretching: use the rescale_intensity() function from scikit-image.exposure
# Plot on the same figure the rescaled image and its histogram
image1 = skio.imread('Hawkes_Bay.jpg')
H1 = ske.histogram(image1, nbins=256)
fig1 = plt.figure(1)
plt.subplot(1, 2, 1), plt.xlim(0, 255), plt.bar(H1[1], H1[0])
plt.subplot(1, 2, 2), plt.imshow(image1), plt.title('Hawkes_Bay')
image2 = ske.rescale_intensity(image1, (100, 200), (25, 250))
H2 = ske.histogram(image2, nbins=256)
fig2 = plt.figure(2)
plt.subplot(1, 2, 1), plt.xlim(0, 255), plt.bar(H2[1], H2[0])
plt.subplot(1, 2, 2), plt.imshow(image2), plt.title(
    'Hawkes_Bay rescale_intensity')

# TO BE COMPLETED

# Histogram equalization: use the equalize_hist() function from the
# scikit-image.exposure
# Plot on the same figure the equalized image and its histogram
# WARNING the equalized histo has abscissa levels between 0 and 1 => multiply by 256
# and apply the floor() operation before the hispogram display
image3 = ske.equalize_hist(image1)
H3 = ske.histogram(image3, nbins=256)
fig3 = plt.figure(3)
plt.subplot(1, 2, 1), plt.xlim(0, 255), plt.bar(H3[1]*256, H3[0]*256)
plt.subplot(1, 2, 2), plt.imshow(image3), plt.title('Hawkes_Bay equalize_hist')

# TO BE COMPLETED
