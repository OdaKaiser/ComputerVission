#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:20:21 2024

@author: nomad
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "rice.png"
image_dir = IMG_SOURCE_DIR + image_file_name
original_image = cv2.imread(image_dir, 0)

def Medianfilter(A,s):
        h, w = A.shape
        B = np.ones((h,w))
        for i in range(0, h - s + 1):
            for j in range(0, w - s + 1):
                sA=A[i:i + s,j:j + s]
                B[i,j]=np.median(sA)
        B=B[0:h-s+1,0:w-s+1]
        return B
    
def Maxfilter(A,s):
        h, w = A.shape
        B = np.ones((h,w))
        for i in range(0, h - s + 1):
            for j in range(0, w - s + 1):
                sA=A[i:i + s,j:j + s]
                B[i,j]=np.max(sA)
        B=B[0:h-s+1,0:w-s+1]
        return B
s=5
Me=Medianfilter(original_image, s)
Me_image=np.array(Me, dtype='uint8') #convert to 8 bit 255 after conv data type is obj
Ma=Maxfilter(original_image, s)
Ma_image=np.array(Ma, dtype='uint8')

plt.figure()
subf1=plt.subplot(1, 3, 1)
subf1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
subf1.set_title('Original')
subf1.axis('off')
subf2=plt.subplot(1, 3, 2)
subf2.imshow(Me_image, cmap='gray')
subf2.set_title('Median')
subf2.axis('off')
subf2=plt.subplot(1, 3, 3)
subf2.imshow(Ma_image, cmap='gray')
subf2.set_title('Max')
subf2.axis('off')
