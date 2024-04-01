#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:20:00 2024

@author: nomad
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_1 = "Tour_Eiffel_2.JPG"

image_1_dir = IMG_SOURCE_DIR + image_file_1


img1 = cv2.imread(image_1_dir)  
#img2 = cv2.imread(image_2_dir) 

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)



sift = cv2.SIFT_create()
kp = sift.detect(img1_gray,None)


img=cv2.drawKeypoints(img1_gray,kp,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img1)
plt.axis("off")