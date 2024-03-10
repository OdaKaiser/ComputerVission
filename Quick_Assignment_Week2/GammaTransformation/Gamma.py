#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:25:29 2024

@author: nomad
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size' :7})
plt.figure(dpi=600)

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "parkavenue.jpg"
image_dir = IMG_SOURCE_DIR + image_file_name

original_image = cv2.imread(image_dir, cv2.COLOR_BGR2RGB)
img = np.array(original_image,'float')

plt.figure(figsize=(3,3), dpi=600)
subf=plt.subplot(3, 3, 1)
subf.set_title ('original')
subf.axis('off')
subf.imshow(original_image)

vals = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]

for i,gamma in enumerate(vals):
    c = round(255/(np.max(img)**gamma), 3)
    image_gamma = np.array(c*(img** gamma), 'uint8')
    subf = plt.subplot(3, 3, i+2)
    subf.set_title ('c=' + str(c))
    subf.imshow(image_gamma)
    subf.axis('off')
