# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:17:39 2020

@author: capliera
"""
import numpy as np
import skimage.io as skio
from os.path import join
from skimage.filters import threshold_otsu
import skimage as sk
import matplotlib.pyplot as plt
import skimage.color as skc
import skimage.exposure as ske
    
plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'book_page_2.jpg'
dirpath = r'C:\Users\ocibo\Documents\traitement image\Images'
filepath = join(dirpath, filename)
print(filepath)

img = skio.imread(filepath)

print("Loaded image has dimensions:", img.shape)


# Binarization, global thresholding

# TO BE COMPLETED
img = skc.rgb2gray(img)
thresh = threshold_otsu(img)

Dim=img.shape
binary=np.zeros((Dim[0],Dim[1]))
for i in range(0,Dim[0]): 
    for j in range(0,Dim[1]):
        if img[i,j] >= thresh:
            binary[i,j]=1
           
H1=ske.histogram(img)
            
fig1=plt.figure(1)
plt.subplot(1,2,1), plt.imshow(img), plt.title('image originale')
plt.subplot(1,2,2), plt.xlim(0,1), plt.bar(H1[1],H1[0]), plt.plot(thresh,0), plt.show()
fig2=plt.figure(2)
plt.subplot(1,1,1), plt.imshow(binary), plt.title('image bnaire')
# Binarization improvment: proposed method

# TO BE COMPLETED

thresh2 = 0.8
binary2=np.zeros((Dim[0],Dim[1]))
for i in range(0,Dim[0]): 
    for j in range(0,Dim[1]):
        if img[i,j] >= thresh2:
            binary2[i,j]=1
            
fig3=plt.figure(3)
plt.subplot(1,1,1), plt.imshow(binary2), plt.title('image bnaire')