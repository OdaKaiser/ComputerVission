#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:11:59 2024

@author: nomad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "sherlock.jpg"
image_dir = IMG_SOURCE_DIR + image_file_name

img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE )
F = np.fft.fft2(img)
D0 = 30

F = np.fft.fftshift(F)
M, N = img.shape
u = np.arange(0,M) - M/2; v = np.arange(0,N) - N/2
[U, V] = np.meshgrid(v, u);
D = np.sqrt(np.power(U,2) + np.power(V,2));
H = 1 - np.exp(-D**2/(2*D0*D0))

G = H*F;
G = np.fft.ifftshift(G)
imgOut = np.real(np.fft.ifft2(G))

fig=plt.figure(dpi=300)
plt.subplot(1, 2, 1)
plt.imshow(img,'gray')
plt.title("Original")
plt.axis('off')



plt.subplot(1, 2, 2)
plt.imshow(imgOut,'gray')
plt.title("Filtered")
plt.axis('off')
