# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 07:58:47 2020

@author: capliera
"""

import numpy as np
import skimage.io as skio
from os.path import join
import skimage.color as skc
import scipy.ndimage as scnd
import skimage.filters as skf
from skimage import color
import math
import skimage as sk
import skimage.data as skd
import skimage.util as sku
import matplotlib.pyplot as plt
 
    
plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'muscle.tif'
dirpath = r'C:\Users\ocibo\Documents\traitement image\Images'
filepath = join(dirpath, filename)
print(filepath)

imgo = skio.imread(filepath)

print("Loaded image has dimensions:", imgo.shape)
imgo = sk.img_as_float(imgo)

img= skd.coffee()
img=skc.rgb2gray(img)
img=sku.img_as_float(img)

# Image gradient components Gx and Gy computation via Sobel filtering

# TO BE COMPLETED

img_sobel_x=img
img_sobel_x=skf.sobel_h(img_sobel_x,mask=None)

img_sobel_y=img
img_sobel_y=skf.sobel_v(img_sobel_y,mask=None)

fig1=plt.figure(1)
plt.subplot(2,2,1), plt.imshow(img), plt.title('image ')
plt.subplot(2,2,2), plt.imshow(img_sobel_x), plt.title('Gradx')
plt.subplot(2,2,3), plt.imshow(img_sobel_y), plt.title('Grady')
# Gradient magnitude and orientation computation 

# TO BE COMPLETED


Dim=img.shape
I_magn=np.zeros((Dim[0],Dim[1]))
for i in range(0,Dim[0]): 
    for j in range(0,Dim[1]):
        I_magn[i,j]=math.sqrt(img_sobel_x[i,j]**2+img_sobel_y[i,j]**2)

I_orient=img_sobel_x/img_sobel_y
I_orient=np.arctan(I_orient)      
fig1
fig2=plt.figure(2)
plt.subplot(2,2,1), plt.imshow(img), plt.title('image ')
plt.subplot(2,2,2), plt.imshow(I_magn), plt.title('magnitude ')
plt.subplot(2,2,3), plt.imshow(I_orient), plt.title('orientation ')
        
#Edge pixels detection : 1st method

# TO BE COMPLETED
B_img=np.zeros((Dim[0],Dim[1]))
max=I_magn.max()

T=0.29*max
for i in range(0,Dim[0]): 
    for j in range(0,Dim[1]):
        if (T<=I_magn[i,j]) and (I_magn[i,j]<=max) :
            B_img[i,j]=I_magn[i,j]
            
fig3=plt.figure(3)
plt.subplot(2,2,1), plt.imshow(img), plt.title('image ')
plt.subplot(2,2,2), plt.imshow(B_img), plt.title('B image ')


# Edge pixels detection : second method with hysteresis thresholding

# TO BE COMPLETED

C_img=np.zeros((Dim[0],Dim[1]))
Tl=0.15*max
Th=0.45*max
for i in range(0,Dim[0]): 
    for j in range(0,Dim[1]):
        if Th<=I_magn[i,j]:
            C_img[i,j]=I_magn[i,j]
        if(j!=0 and i!=0):
            if Tl<=I_magn[i,j]<Th:
                if(I_magn[i+1,j]>=Th) or (I_magn[i-1,j]>=Th)  or (I_magn[i,j+1]>=Th) or (I_magn[i,j-1]>=Th):
                    C_img[i,j]=I_magn[i,j]
            
fig3=plt.figure(3)
plt.subplot(2,2,3), plt.imshow(C_img), plt.title('C image ')
