#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:36:56 2024

@author: nomad
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv(A,k):
    kh, kw = k.shape
    h, w = A.shape
    B = np.ones((h,w))
    for i in range(0, h - kh -1):
        for j in range(0, w - kw - 1):
            sA = A[i:i + kh, j:j + kw]
            B[i,j] = np.sum(k*sA)
    B=B[0:h-kh+1,0:w-kw+1]
    return B

def Gaussian(l,sigma):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l) 
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel 

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "SkeletonXray.png"
image_dir = IMG_SOURCE_DIR + image_file_name
original_image = cv2.imread(image_dir, 0)

l = 5
sigma = 1.0
k = Gaussian(l, sigma)
B = conv(original_image,k)
imgB=np.array(B, dtype='uint8')

plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(imgB, cmap='gray')
plt.axis('off')
plt.show()
