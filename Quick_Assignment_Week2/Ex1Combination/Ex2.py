#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:23:19 2024

@author: nomad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "SkeletonXray.png"
image_dir = IMG_SOURCE_DIR + image_file_name

#image original
original_image_a = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
plt.imshow(original_image_a, cmap='gray')
plt.axis('off')
plt.show()

#image a
# Calculation of Sobelx Sobel(gray/image, ddepth=cv2.CV_32F/CV_8U, dx=1/0, dy=1/0, ksize=ksize)
sobelx = cv2.Sobel(original_image_a,cv2.CV_64F,1,0,ksize=5) 





# Calculation of Sobely 
sobely = cv2.Sobel(original_image_a,cv2.CV_64F,0,1,ksize=5)


#image b
sum_sobel_of_a_d = cv2.add(sobelx, sobely)
plt.imshow(sum_sobel_of_a_d, cmap='gray')
plt.axis('off')
plt.show()


