# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:08:26 2020

@author: capliera
"""

import numpy as np
import scipy.fftpack as sc
import skimage.io as skio
from os.path import join

import skimage as sk
from matplotlib import cm
import matplotlib.pyplot as plt
 
    
plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

#___________________________________________________________________________
def filtre_homorph(taille,rh=3.0, rl=0.5, delta=0.5,fig=0):
    s=(taille[0],taille[1])
    H=np.zeros(s);
    [U, V]=np.ogrid[-0.5:0.5:1.0/taille[0],-0.5:0.5:1.0/taille[1]];
    H= (rh-rl)*(1-np.exp(-(U*U+V*V)/(2*delta*delta)))+rl
    
    U1=np.copy(U[::10,::10])
    V1=np.copy(V[::10])
    H1=np.copy(H[::10])

    if fig>0:
        
        fig = plt.figure(fig)
        ax = fig.gca(projection='3d')
        ax.plot_surface(U1,V1,H1, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(0, 2)       
        plt.title('High pass filter transfer function')

    return H
 #____________________________________________________________________________
    
filename = r'tunnel.jpg'
dirpath = r'..\..\images'
filepath = join(dirpath, filename)
print(filepath)

I0=skio.imread(filepath)
print("Loaded image has dimensions:", I0.shape)
print("Loaded values are of type:", I0.dtype)
I=sk.img_as_float(I0)
plt.figure(1)
plt.imshow(I0), plt.title('Original Image')

# Implement the homomorphic filtering process and display the filtered image

# TO BE COMPLETED

