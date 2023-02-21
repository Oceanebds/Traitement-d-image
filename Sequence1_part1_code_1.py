# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:56:53 2020

@author: capliera


"""

import math
import numpy as np
import skimage.io as skio
import skimage.data as skd
import skimage.color as skc
import matplotlib.pyplot as plt
import skimage.filters as skf
import skimage.util as sku
import skimage.metrics as skme
import skimage.morphology as skmo
from skimage.filters.rank import mean_bilateral
from os.path import join
from skimage.morphology import square

import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt


import skimage.exposure as ske


#All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and 
# the image figure size

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

# Several test images are available in the scikit-images.data module.
# Those images are entiltled: coins, astronaut (color), brick, camera, cell,
# chekerboard, chelsea (color), clock, coffee (color), colorwheel (color), grass (color),
# gravel (color), horse (black and white), moon, retina (color), rocket (color).

# To download any of those images just use skd.image_name()
# To download any other image, use skio.imread('image_name')


## Grey level image reading and information

I=skd.coins()
Dim = I.shape
nb_pixels=I.size
Max_grey_level=np.max(I)
Min_grey_level=np.min(I)
Mean_grey_level=np.mean(I)

# Image display
fig1=plt.figure(1)                    
skio.imshow(I)
plt.title('Coins')


# Color image reading and RGB channel visualization
J=skd.coffee()
fig4=plt.figure(4)
plt.subplot(2,2,1), plt.imshow(J), plt.title('Color img')
plt.subplot(2,2,2), plt.imshow(J[:,:,0]), plt.title('Red Channel')
plt.subplot(2,2,3), plt.imshow(J[:,:,1]), plt.title('Green Channel')
plt.subplot(2,2,4), plt.imshow(J[:,:,2]), plt.title('Blue Channel')
 
#luminance

L=(0.3*J[:,:,0]+0.58*J[:,:,1]+0.11*J[:,:,2])
fig5=plt.figure(5)
plt.subplot(2,2,1), plt.imshow(L), plt.title('Color img')

#ligne 200

Extract=L[200,:]
fig6=plt.figure(6)
plt.subplot(2,2,3), plt.plot(Extract), plt.title('Luminance ligne 200')


image3=skio.imread('Fruits.bmp')
H=ske.histogram(image3,nbins=256)
fig7=plt.figure(7)

plt.subplot(1,3,1), plt.xlim(0,255), plt.bar(H[1],H[0])
plt.title('Histogramm')

plt.subplot(1,2,2),skio.imshow(image3)
plt.title('Image') 
#plt.subplot(1,3,3),plt.imshow(image3)
#plt.title('Display with matplotlib')
# Luminance extraction : compute the luminance as 0.3*R+0.58*G+0.11*B
# Display the luninance evolution along line 200

# WARNING : instructon A=B does not create a new array, A and B are the same 
# object then modifying A will also modify B => use .copy() function in order
# duplicate an image

#TO BE COMPLETED

#Two imshow() functions for image display
# Compute the histogram of image 3 (cf. ske.histogram())
#Display the hitogram ((cf. plt.bar())

#image3=skio.imread('CH0SRC.TIF')

#TO BE COMPLETED

# Display image3 by suing first skio.image() and second by usig plt.imshow()
#What is the difference between both functions ?      

# TO BE COMPLETED

