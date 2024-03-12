#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:27:58 2024

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


IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "SkeletonXray.png"
image_dir = IMG_SOURCE_DIR + image_file_name
original_image = cv2.imread(image_dir, 0)

k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
B = conv(original_image,k)
imgB=np.array(B, dtype='uint8')

plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(imgB, cmap='gray')
plt.axis('off')
plt.show()
