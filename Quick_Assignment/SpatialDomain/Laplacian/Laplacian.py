#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:56:26 2024

@author: nomad
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Negative(image):

    height, width, channel = image.shape 
     
    for i in range(0, height - 1): 
        for j in range(0, width - 1): 
            pixel = image[i, j] 
            pixel[0] = 255 - pixel[0]       
            pixel[1] = 255 - pixel[1] 
            pixel[2] = 255 - pixel[2] 
            image[i, j] = pixel 
            
    return image

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "moon.tif"
image_dir = IMG_SOURCE_DIR + image_file_name

original_image = cv2.imread(image_dir)
plt.imshow(original_image)
plt.axis('off')
plt.show()

gblur = cv2.GaussianBlur(original_image, (3,3), 0)
plt.imshow(original_image)
plt.axis('off')
plt.show()

laplacian_kernel_a = np.array([[0, 1, 0],
                             [1, -4, 1], 
                             [0, 1, 0]])

laplacian_kernel_b = np.array([[1, 1, 1],
                               [1, -8, 1], 
                               [1, 1, 1]])


result_a = cv2.filter2D(gblur, -1, laplacian_kernel_a)
plt.imshow(result_a)
plt.axis('off')
plt.show()

result_a = Negative(result_a)
plt.imshow(result_a)
plt.axis('off')
plt.show()


result_b = cv2.filter2D(gblur, -1, laplacian_kernel_a)
plt.imshow(result_b)
plt.axis('off')
plt.show()

result_b = Negative(result_b)
plt.imshow(result_b)
plt.axis('off')
plt.show()

