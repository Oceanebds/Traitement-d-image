# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:34:03 2022

@author: ocibo
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


#All figures closing
plt.close('all')

image = skd.coffee()
image=skc.rgb2gray(image)
image=sku.img_as_float(image)
fig1=plt.figure(1)
plt.subplot(2,2,1), plt.imshow(image), plt.title('coffee')
print("Loaded image has dimensions:", image.shape)
print("Loaded values are of type:", image.dtype)

image_noise=sku.random_noise(image, mode='gaussian')

fig1=plt.figure(1)
plt.subplot(2,2,2), plt.imshow(image_noise), plt.title('image with gaussian noise')
Dimm=image.shape


PSNR=skme.peak_signal_noise_ratio(image, image_noise)

print("PSNRcalc= ",PSNR)

# Add to the image a Gaussian noise of variance 0.01 by using the random.noise 
# function coming from skimage.util and compute the psnr between the original
# and the noisy images by using the peak_signal_noise_ratio funtion from
# skimage.metrics

# TO BE COMPLETED
# 
PNSR_vect=np.empty( 25, dtype = float)
for tau in range(10,260,10):
    k=0
    c=0
    somme1=0
    somme2=0
    val=1.2
    I_new6=image
    for i in range(1,Dimm[0]-1):
        for j in range(1,Dimm[0]-1):
            somme1=0
            somme2=0
            for a in range(2):
                for b in range(2):
                    if abs(image[i,j]-image[i-a,j-b]) <tau:
                        c=1
                    else:
                        c=0
                    somme1 += c*image[i-a,j-b]
                    somme2+=c
        
            val=somme1/somme2
            I_new6[i,j]=val
            
    PNSR_vect[k]=skme.peak_signal_noise_ratio(image, I_new6)
    k+=1


x=np.arange(1,26)
figure2=plt.figure(2)
plt.plot(x, PNSR_vect, label = 'graphe' )

tau=20
c=0
somme1=0
somme2=0
val=1.2
I_new7=image
for i in range(1,Dimm[0]-1):
    for j in range(1,Dimm[0]-1):
        somme1=0
        somme2=0
        for a in range(2):
            for b in range(2):
                if abs(image[i,j]-image[i-a,j-b]) <tau:
                    c=1
                else:
                    c=0
                somme1 += c*image[i-a,j-b]
                somme2+=c
        val=somme1/somme2
        I_new7[i,j]=val
        
tau=220
c=0
somme1=0
somme2=0
I_new8=image
for i in range(1,Dimm[0]-1):
    for j in range(1,Dimm[0]-1):
        somme1=0
        somme2=0
        for a in range(2):
            for b in range(2):
                if abs(image[i,j]-image[i-a,j-b]) <tau:
                    c=1
                else:
                    c=0
                somme1 += c*image[i-a,j-b]
                somme2+=c
        val=somme1/somme2
        I_new8[i,j]=val

fig3=plt.figure(3)
plt.subplot(2,2,1), plt.imshow(I_new7), plt.title('tau=20')
plt.subplot(2,2,2), plt.imshow(I_new8), plt.title('tau=220')
        
# fig4=plt.figure(4)
# plt.subplot(2,2,1), plt.imshow(I_new6), plt.title('Image filtree')



#Adaptive filtering - Determining the best tau value
# Implement the adaptive filtering for different tau values
# Determine the best tau value by computing the psnr between the original
# and the adaptive filtered images - Display the filtered image corresponding
# to the best tau value

# TO BE COMPLETED

# Bilateral filtering - Use the mean_bilateral function in order to filter
# the noisy image and compute the psnr between the origainal and the bilateral
# filtered images

# TO BE COMPLETED