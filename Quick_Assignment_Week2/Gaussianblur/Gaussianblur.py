#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:36:56 2024

@author: nomad
"""
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):



def convolution(image, kernel, average=False, verbose=False):
 

 



IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "strawberries.jpg"
image_dir = IMG_SOURCE_DIR + image_file_name
original_image = cv2.imread(image_dir)
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 



plt.imshow(gaussian_image)
plt.axis('off')
plt.show()