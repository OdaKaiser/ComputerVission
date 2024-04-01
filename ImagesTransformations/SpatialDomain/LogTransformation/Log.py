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
maxV = 255/np.log(1+np.max(original_image))
vals = np.linspace(0, maxV, 8, dtype=int)
plt.figure(figsize=(3,3), dpi=600)
subf=plt.subplot(3, 3, 1)
subf.set_title ('original')
subf.axis('off')
subf.imshow(original_image)

for i,c in enumerate(vals):
    log_image = c*(np.log(img+1))
    log_image = np.array(log_image, dtype = 'uint8')
    subf = plt.subplot(3, 3, i+2)
    subf.set_title ('c=' + str(c))
    subf.imshow(log_image)
    subf.axis('off')

        