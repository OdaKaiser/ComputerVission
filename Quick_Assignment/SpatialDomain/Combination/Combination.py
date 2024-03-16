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

#image d
grad_x = cv2.Sobel(original_image_a, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(original_image_a, cv2.CV_64F, 0, 1, ksize=3)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
gradient_magnitude_d = np.sqrt(abs_grad_x**2 + abs_grad_y**2)
plt.imshow(gradient_magnitude_d, cmap='gray')
plt.axis('off')
plt.show()

#image e
# Calculation of Sobelx Sobel(gray/image, ddepth=cv2.CV_32F/CV_8U, dx=1/0, dy=1/0, ksize=ksize)\
gradient_magnitude_casted_d=np.array(gradient_magnitude_d, dtype='uint8')
sobelx = cv2.Sobel(gradient_magnitude_casted_d,cv2.CV_64F,1,0,ksize=5) 
# Calculation of Sobely 
sobely = cv2.Sobel(gradient_magnitude_casted_d,cv2.CV_64F,0,1,ksize=5)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sum_sobel_of_a_e = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
plt.imshow(sum_sobel_of_a_e, cmap='gray')
plt.axis('off')
plt.show()

#Image f
image_f = original_image_a*sum_sobel_of_a_e
plt.imshow(image_f, cmap='gray')
plt.axis('off')
plt.show()

#image g
image_g = cv2.add(original_image_a, image_f)
plt.imshow(image_g, cmap='gray')
plt.axis('off')
plt.show()

#image h, s = c*rγ
image_h = np.array(255*(image_g/255)**2.2,dtype='uint8')
plt.imshow(image_h, cmap='gray')
plt.axis('off')
plt.show()

