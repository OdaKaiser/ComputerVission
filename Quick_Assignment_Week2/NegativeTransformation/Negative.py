#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:25:29 2024

@author: nomad
"""
import cv2
import matplotlib.pyplot as plt

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "parkavenue.jpg"
image_dir = IMG_SOURCE_DIR + image_file_name

# debug
# =============================================================================
# print(image)
# =============================================================================

original_image = cv2.imread(image_dir, cv2.IMREAD_COLOR)
image_negative = original_image

height, width, channel = original_image.shape

#debug
# =============================================================================
# print(height)
# print(width)
# print(channel)
# =============================================================================

for i in range(0, height - 1):
    for j in range(0, width - 1):
        #get idividual pixel
        pixel = original_image[i, j]
        
        pixel[0] = 255 - pixel[0] #R
        pixel[1] = 255 - pixel[1] #G
        pixel[2] = 255 - pixel[2] #B
        
        #Assign to neagtive variable
        image_negative[i, j] = pixel

#Display
plt.figure()
subf1=plt.subplot(1, 2, 1)
subf1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
subf1.set_title('Original')
subf1.axis('off')
subf2=plt.subplot(1, 2, 2)
subf2.imshow(cv2.cvtColor(image_negative, cv2.COLOR_BGR2RGB))
subf2.set_title('Negative')
subf2.axis('off')
plt.show()
        