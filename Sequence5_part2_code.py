# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:08:23 2020

@author: capliera
"""

from sklearn import cluster
import skimage.io as skio
from os.path import join
import matplotlib.pyplot as plt
    
plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'leopard.jpg'
dirpath = r'..\..\images'
filepath = join(dirpath, filename)
print(filepath)

img = skio.imread(filepath)

print("Loaded image has dimensions:", img.shape)
plt.figure(), plt.imshow(img, cmap='gray')

# For question 2.2 implemnt here the computation of Gabor features
# Use the gabor() and the gauss() functions to be imported from skimage.filters module
# TO BE COMPLETED

# k-means clustering of the image
X = img.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(X)

# extract means of each cluster & clustered population
clusters_means = k_means.cluster_centers_.squeeze()
X_clustered = k_means.labels_
print('# of Observations:', X.shape)
print('Clusters Means:', clusters_means)

# Display the clustered image
X_clustered.shape = img.shape
plt.figure(), plt.imshow(X_clustered), plt.title('Kmean segmentation')