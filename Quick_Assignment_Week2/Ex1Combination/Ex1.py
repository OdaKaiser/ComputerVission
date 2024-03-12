#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:42:56 2024

@author: nomad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "SkeletonXray.png"
image_dir = IMG_SOURCE_DIR + image_file_name

#image a
original_image_a = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
plt.imshow(original_image_a, cmap='gray')
plt.axis('off')
plt.show()

#image b
laplace_image_b = cv2.Laplacian(original_image_a, cv2.CV_64F) 
plt.imshow(laplace_image_b, cmap='gray')
plt.axis('off')
plt.show()

#image c
img_b_tmp = np.array(laplace_image_b, 'uint8')
sum_image_ab_c = cv2.add(original_image_a, img_b_tmp)
plt.imshow(sum_image_ab_c, cmap='gray')
plt.axis('off')
plt.show()

# Calculation of Sobelx Sobel(gray/image, ddepth=cv2.CV_32F/CV_8U, dx=1/0, dy=1/0, ksize=ksize)
sobelx = cv2.Sobel(original_image_a,cv2.CV_64F,1,0,ksize=5) 
# =============================================================================
# plt.imshow(sobelx, cmap='gray')
# plt.axis('off')
# plt.show()  
# =============================================================================


# Calculation of Sobely 
sobely = cv2.Sobel(original_image_a,cv2.CV_64F,0,1,ksize=5)
# =============================================================================
# plt.imshow(sobely, cmap='gray')
# plt.axis('off')
# plt.show()  
# =============================================================================
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
#image d
sum_sobel_of_a_d = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
plt.imshow(sum_sobel_of_a_d, cmap='gray')
plt.axis('off')
plt.show()


