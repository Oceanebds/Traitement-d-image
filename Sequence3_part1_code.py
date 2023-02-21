# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:56:10 2020

@author: capliera
"""

import numpy as np
import scipy.fftpack as sc
import skimage.io as skio
import skimage.color as skc
import skimage.data as skd
import skimage.filters as skf
import matplotlib.pyplot as plt
from os.path import join
from skimage.util import img_as_float
from matplotlib import cm
import skimage.util as sku
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

# Transfer function of an ideal low pass filter
def filtrePB_Porte(fc,taille):
    s=(taille[0],taille[1])
    H=np.zeros(s);
    [U, V]=np.ogrid[-0.5:0.5:1.0/taille[0],-0.5:0.5:1.0/taille[1]];
    H = np.sqrt(U*U+V*V)<fc;
    #fc est le rayon de coupure
    return H  

def filtrebutterworth(fc,taille,ordre=1,fig=0):
    s=(taille[0],taille[1])
    H=np.zeros(s);
    [U, V]=np.ogrid[-0.5:0.5:1.0/taille[0],-0.5:0.5:1.0/taille[1]];
    H =1.0/ (1.0+0.414*np.power(np.sqrt(U*U+V*V)/(fc),2*ordre));
    
    U1=np.copy(U[::10,::10])
    V1=np.copy(V[::10])
    H1=np.copy(H[::10])
    if fig>0:
        
        fig = plt.figure(fig)
        ax = fig.gca(projection='3d')
        ax.plot_surface(U1,V1,H1, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(0, 1.01)

        stringfc=np.str(fc)
        stringordre=np.str(ordre)
        plt.title('Butterworth filter fc='+stringfc+', ordre='+stringordre)

    return H

#Image loadind and display
# Plot in the same figure the image and its Fourier transform

filename = r'cameraman.bmp'
dirpath = r'..\images'
filepath = join(dirpath, filename)
print(filepath)

image_coffee= skd.coffee()
image_coffee=skc.rgb2gray(image_coffee)
image_coffee=sku.img_as_float(image_coffee)


TFI2=sc.fft2(image_coffee)
TFI2=sc.fftshift(TFI2)
spectre2=np.abs(TFI2**2)

fig=plt.figure(figsize=(12,8))                    
plt.subplot(1,2,1), plt.imshow(image_coffee)
plt.subplot(1,2,2),plt.imshow(np.log10(1+spectre2))

fig4=plt.figure(4)
plt.subplot(1,2,1), plt.imshow(filtrePB_Porte(0.05, image_coffee.shape))

filtrebutterworth(0.05,image_coffee.shape,ordre=5,fig=5)

image=skio.imread(filepath)
print("Loaded image has dimensions:", image.shape)
print("Loaded values are of type:", image.dtype)

# Image Fourier Transform
TFI=sc.fft2(image)
TFI=sc.fftshift(TFI)
spectre=np.abs(TFI**2)

fig=plt.figure(figsize=(12,8))                    
plt.subplot(1,2,1), plt.imshow(image)
plt.subplot(1,2,2),plt.imshow(np.log10(1+spectre))

#Low pass and band pass Frequential filtering on coffee image

# display the coffee image and its spectrum

# TO BE COMPLETED

# Display the ideal low pass filter transfer functions
# taille=np.shape(J)

# Filtre2=filtrePB_Porte(0.05,taille)
# plt.subplot(2,2,3), plt.imshow(Filtre2),plt.title('Low pass filter fc=0.05')

# Filter the coffee image in the frequency domain, display the filtered 
# spectrum Low after low pass filtering and the obtained image

#TO BE COMPLETED

# Comparison with a Butterworth filter

# Generate several transfer functions of Buetterworth filters with different 
# parameters. Comment

#TO BE COMPLETE

# Filter the coffee image in the frequency domain with a Butterworht filter

#TO BE COMPLETE