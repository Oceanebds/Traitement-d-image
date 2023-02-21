# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:03:51 2020

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



#All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and 
# the image figure size

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

# Image loading and display
filename = r'House.jpg'
#filename = r'Train.jpg'
dirpath = r'C:\Users\ocibo\Documents\traitement image\sequence 2'
filepath = join(dirpath, filename)
print(filepath)


image=skio.imread(filepath)
fig1=plt.figure(1)
plt.subplot(2,2,1), plt.imshow(image), plt.title(filename)
print("Loaded image has dimensions:", image.shape)
print("Loaded values are of type:", image.dtype)

# Image loading and display
filename2 = r'Train.jpg'
#filename = r'Train.jpg'
dirpath = r'C:\Users\ocibo\Documents\traitement image\sequence 2'
filepath2 = join(dirpath, filename2)
print(filepath2)


image2=skio.imread(filepath2)
fig2=plt.figure(2)
plt.subplot(2,2,1), plt.imshow(image2), plt.title(filename2)
print("Loaded image has dimensions:", image.shape)
print("Loaded values are of type:", image.dtype)

# Appropriate filtering with different neighborhood sizes in order to remove 
# salt and pepper noise

#MATRICE 3x3
I_new1=image
mat3=np.zeros((3,3))
Dim=image.shape
for i in range(1,Dim[0]-1): #on exclut les bords
    for j in range(1,Dim[1]-1):
        mat3[0,0]=image[i-1,j-1]
        mat3[0,1]=image[i-1,j]
        mat3[0,2]=image[i-1,j+1]
        mat3[1,0]=image[i,j-1]
        mat3[1,1]=image[i,j]
        mat3[1,2]=image[i,j+1]
        mat3[2,0]=image[i+1,j-1]
        mat3[2,1]=image[i+1,j]
        mat3[2,2]=image[i+1,j+1]
        #mat3=image[i-1:i+1,j-1:j+1]
        I_new1[i,j]=(1/9)*np.sum(mat3)
        
fig1=plt.figure(1)
plt.subplot(2,2,2), skio.imshow(I_new1), plt.title('Image filtree en 3x3')

I_new2=image2
mat3_2=np.zeros((3,3))
Dim2=image2.shape
for i in range(1,Dim2[0]-1): #on exclut les bords
    for j in range(1,Dim2[1]-1):
        mat3_2[0,0]=image2[i-1,j-1]
        mat3_2[0,1]=image2[i-1,j]
        mat3_2[0,2]=image2[i-1,j+1]
        mat3_2[1,0]=image2[i,j-1]
        mat3_2[1,1]=image2[i,j]
        mat3_2[1,2]=image2[i,j+1]
        mat3_2[2,0]=image2[i+1,j-1]
        mat3_2[2,1]=image2[i+1,j]
        mat3_2[2,2]=image2[i+1,j+1]
        #mat3_2=image2[i-1:i+1,j-1:j+1]
        I_new2[i,j]=(1/9)*np.sum(mat3_2)
        
fig2=plt.figure(2)
plt.subplot(2,2,2), plt.imshow(I_new2), plt.title('Image filtree en 3x3')

#MATRICE 7x7
mat7=np.zeros((7,7))
I_new3=image
for i in range(3,Dim[0]-3): #on exclut les bords
    for j in range(3,Dim[1]-3):
        mat7=image[i-3:i+3,j-3:j+3]
        # for a in range(7):
        #     for b in range(7):
        #         I_new3[i,j]=image[i-a,j-b]
        I_new3[i,j]=(1/49)*np.sum(mat7)

fig1=plt.figure(1)
plt.subplot(2,2,3), plt.imshow(I_new3), plt.title('Image filtree en 7x7')

mat7_2=np.zeros((7,7))
I_new4=image2
for i in range(3,Dim2[0]-3): #on exclut les bords
    for j in range(3,Dim2[1]-3):
        mat7_2=image2[i-3:i+3,j-3:j+3]
        I_new4[i,j]=(1/49)*np.sum(mat7_2)

fig2=plt.figure(2)
plt.subplot(2,2,3), plt.imshow(I_new4), plt.title('Image filtree en 7x7')

#MATRICE 13x13
mat13=np.zeros((13,13))
I_new5=image
for i in range(6,Dim[0]-6): #on exclut les bords
    for j in range(6,Dim[1]-6):
        mat13=image[i-6:i+6,j-6:j+6]
        # for a in range(7):
        #     for b in range(7):
        #         I_new3[i,j]=image[i-a,j-b]
        I_new5[i,j]=(1/169)*np.sum(mat13)
        
fig1=plt.figure(1)
plt.subplot(2,2,4), plt.imshow(I_new5), plt.title('Image filtree en 13x13')


mat13_2=np.zeros((13,13))
I_new6=image2
for i in range(6,Dim2[0]-6): #on exclut les bords
    for j in range(6,Dim2[1]-6):
        mat13_2=image2[i-6:i+6,j-6:j+6]
        # for a in range(7):
        #     for b in range(7):
        #         I_new3[i,j]=image[i-a,j-b]
        I_new6[i,j]=(1/169)*np.sum(mat13_2)
        
fig2=plt.figure(2)
plt.subplot(2,2,4), plt.imshow(I_new6), plt.title('Image filtree en 13x13')
# TO BE COMPLETED

# Adaptive filtering

# Image loading and display

# image = skd.coffee()
# image=skc.rgb2gray(image)
# image=sku.img_as_float(image)
# fig3=plt.figure(3)
# plt.subplot(2,2,1), plt.imshow(image), plt.title(filename)
# print("Loaded image has dimensions:", image.shape)
# print("Loaded values are of type:", image.dtype)

# image_noise=sku.random_noise(image, mode='gaussian')

# fig3=plt.figure(3)
# plt.subplot(2,2,2), plt.imshow(image_noise), plt.title('image with gaussian noise')
# Dimm=image.shape


# PSNR=skme.peak_signal_noise_ratio(image, image_noise)

# print("PSNRcalc= ",PSNR)

# # Add to the image a Gaussian noise of variance 0.01 by using the random.noise 
# # function coming from skimage.util and compute the psnr between the original
# # and the noisy images by using the peak_signal_noise_ratio funtion from
# # skimage.metrics

# # TO BE COMPLETED
# # 
# PNSR_vect=np.arange(25)
# for tau in range(10,250,10):
#     k=0
#     c=0
#     somme1=0
#     somme2=0
#     I_new6=image
#     for i in range(1,Dimm[0]-1):
#         for j in range(1,Dimm[0]-1):
#             somme1=0
#             somme2=0
#             for a in range(2):
#                 for b in range(2):
#                     if (image[i,j]-image[i-a,j-b] <tau):
#                         c=1
#                     else:
#                         c=0
#                     somme1 += c*image[i-a,j-b]
#                     somme2+=c
        
#             I_new6[i,j]=somme1/somme2
            
#     PNSR_vect[k]=skme.peak_signal_noise_ratio(image, I_new6)
#     k+=1

# fig4=figure(4)
# x=np.arange(1,26)
# plt.plot(x, PNSR_vect, label = 'graphe' )
# plt.show()
        
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







