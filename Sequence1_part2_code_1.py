# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:22 2020

@author: capliera
"""

import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt

import skimage.exposure as ske

def quantize(im, levels):
    """
    Function to run uniform gray-level gray-scale Quantization.
    This takes in an image, and buckets the gray values depending on the params.
    Args:
        im (array): image to be quantized as an array of values from 0 to 255
        levels (int): number of grey levels to quantize to.          
    Return:
        the quantized image
    """
   
    # get int type
    dtype = im.dtype    
    returnImage = np.floor((im/(256/float(levels))))

    print(returnImage)
    return np.array(returnImage, dtype)


#All figures closing
plt.close('all')
# matplotlib for gray level images display: fix the colormap and 
# the image figure size
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

## Grey level image uniform quantization

I_256=skio.imread('alumgrns.tif')
# Image display
fig1=plt.figure(1)                  
skio.imshow(I_256)
plt.subplot(1,1,1), plt.imshow(I_256), plt.title('256 grey levels')
axes = plt.gca() 
axes.axis('off') 

#Image quantization with different numbers of grey levels

# TO BE COMPLETED
I_fruits=skio.imread('fruits.bmp')

fig2=plt.figure(2)                  
skio.imshow(I_fruits)
plt.subplot(1,4,1), plt.imshow(I_fruits), plt.title('image origianle fruits')
axes = plt.gca() 
axes.axis('off') 

fig3=plt.figure(3)
plt.subplot(2,2,1), plt.imshow(quantize(I_fruits, 256)), plt.title('256 fruits')
plt.subplot(2,2,2), plt.imshow(quantize(I_fruits, 128)), plt.title('128 fruits')
plt.subplot(2,2,3), plt.imshow(quantize(I_fruits, 64)), plt.title('64 fruits')
plt.subplot(2,2,4), plt.imshow(quantize(I_fruits, 32)), plt.title('32 fruits')
fig4=plt.figure(4)
plt.subplot(2,2,1), plt.imshow(quantize(I_fruits, 16)), plt.title('16 fruits')
plt.subplot(2,2,2), plt.imshow(quantize(I_fruits, 8)), plt.title('8 fruits')
plt.subplot(2,2,3), plt.imshow(quantize(I_fruits, 4)), plt.title('4 fruits')
plt.subplot(2,2,4), plt.imshow(quantize(I_fruits,2)), plt.title('2 fruits')
    
#two different images having the same histogramm

I_new1=np.zeros((256,256),dtype='uint8')
I_new2=np.zeros((256,256),dtype='uint8')

for i in range(256):
    for j in range(256):
        I_new1[i,j]=256-i
        
for i in range(256):
    for j in range( 256):
        I_new2[i,j]=256-j
        
fig5=plt.figure(5)
H1=ske.histogram(I_new1,nbins=256)
plt.subplot(1,2,1), plt.xlim(0,255), plt.bar(H1[1],H1[0])
plt.subplot(1,2,2), plt.imshow(I_new1), plt.title('I_new1')

fig6=plt.figure(6)
H2=ske.histogram(I_new2,nbins=256)
plt.subplot(1,2,1), plt.xlim(0,255), plt.bar(H2[1],H2[0])
plt.subplot(1,2,2), plt.imshow(I_new2), plt.title('I_new2')

#neighborhood impact

I_new3=np.zeros((256,256),dtype='uint8')
I_new4=np.zeros((256,256),dtype='uint8')

for i in range(256):
    for j in range(256):
        I_new3[i,j]=100


for i in range(256):
    for j in range(256):
        I_new4[i,j]=170
        
        
for i in range(100,156):
    for j in range(100,156):
        I_new3[i,j]=135
        
for i in range(100,156):
    for j in range(100,156):
        I_new4[i,j]=135
        
fig7=plt.figure(7)
plt.subplot(1,2,1), skio.imshow(I_new3), plt.title('I_new3')
plt.subplot(1,2,2), skio.imshow(I_new4), plt.title('I_new4')

#5

I_new5=np.zeros((256,256),dtype='uint8')

for i in range(48,208):
    for j in range(48,208):
        if ((i-128)**2+(j-128)**2 < 6400):
            I_new5[i,j]=1
            
fig8=plt.figure(8)
plt.subplot(1,1,1), plt.imshow(I_new5), plt.title('I_new5')

#combine both
I_new6=np.ones((256,256),dtype='uint8')

for i in range(48,208):
    for j in range(48,208):
        if ((i-128)**2+(j-128)**2 < 6400):
            I_new6[i,j]=256-j

            
fig9=plt.figure(9)
plt.subplot(1,1,1), plt.imshow(I_new6), plt.title('I_new6')
        
        








