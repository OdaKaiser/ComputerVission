import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 
  
# Open the image. 
IMG_SOURCE_DIR = "/home/nomad/Workspace/Master_WorkSpace/ComputerVision/SamplePics/"
image_file_name = "sample.jpg"
image_dir = IMG_SOURCE_DIR + image_file_name
original_image = cv2.imread(image_dir)
  
# Define parameters. 
r1 = 50
s1 = 0
r2 = 160
s2 = 200
  
# Vectorize the function to apply it to each value in the Numpy array. 
pixelVal_vec = np.vectorize(pixelVal) 
  
# Apply contrast stretching. 
contrast_stretched = pixelVal_vec(original_image, r1, s1, r2, s2) 

#Display
plt.figure()
subf1=plt.subplot(1, 2, 1)
subf1.imshow(original_image)
subf1.set_title('Original')
subf1.axis('off')
subf2=plt.subplot(1, 2, 2)
subf2.imshow(contrast_stretched)
subf2.set_title('Pisewise')
subf2.axis('off')
plt.show()