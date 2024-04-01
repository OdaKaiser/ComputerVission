import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

# Function to apply SIFT 
def applySIFT(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypointsImage, descriptorImage = sift.detectAndCompute(image,None)
    return keypointsImage, descriptorImage

# Function to get good matches, given feature descriptors
def getGoodMatches(descriptorImage1, descriptorImage2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptorImage1, descriptorImage2, k=2)
    goodList = []
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodList.append([m])
            good.append(m)
            
    return goodList, good
# Function to get Inliers after RANSAC, accepts mask returned
def getInliers(mask, num=10):
    matchesMask = mask.ravel().tolist()
    indices = []
    for ind in range(len(matchesMask)):
        if matchesMask[ind] == 1:
            indices.append(ind)
    matchesMask = [0]*len(matchesMask)
    np.random.shuffle(indices)
    indices = indices[:num]
    for ind in indices:
            matchesMask[ind] = 1
    return matchesMask


# Function to get left most and top most edge points, to translate warped image
def getExtremePoints(image1CornersPlane2):
    xMin = min(image1CornersPlane2[0][0], image1CornersPlane2[1][0])
    yMin = min(image1CornersPlane2[0][1], image1CornersPlane2[3][1])
    xMax = max(image1CornersPlane2[2][0], image1CornersPlane2[3][0])
    yMax = max(image1CornersPlane2[1][1], image1CornersPlane2[2][1])
    return xMin, yMin, xMax, yMax


# Reading images
image1=cv2.imread('Images/a1.jpg')
image2=cv2.imread('Images/a2.jpg')

keypointsImage1, descriptorImage1 = applySIFT(image1)
keypointsImage2, descriptorImage2 = applySIFT(image2)

# Writing the matches detected in the 2 images to the filesystem
Image1Keypoints=cv2.drawKeypoints(image1,keypointsImage1,None)
cv2.imwrite('Results/task1_sift2.jpg',Image1Keypoints)
Image2Keypoints=cv2.drawKeypoints(image2,keypointsImage2,None)
cv2.imwrite('Results/task1_sift1.jpg',Image2Keypoints)

# Get good matches using KNN algorithm between kepoint descriptors 
goodList, good = getGoodMatches(descriptorImage1, descriptorImage2)

# Plotting knn matches based on the keypoint distances computed
imagePlot = cv2.drawMatchesKnn(image1,keypointsImage1,image2,keypointsImage2,goodList,None,flags=2)
cv2.imwrite('Results/task1_matches_knn.jpg',imagePlot)

# Getting keypoint locations as an array of (x,y) coordinates
ptsImage1 = np.array([ keypointsImage1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
ptsImage2 = np.array([ keypointsImage2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

# Getting homography matrix after applying RANSAC on
H, mask = cv2.findHomography(ptsImage1, ptsImage2, cv2.RANSAC)
print('Homography Matrix:')
print(H)
 
# Get 10 inlier matches after applying RANSAC
matchesMask = getInliers(mask, 10)
inlierImage = cv2.drawMatches(image1,keypointsImage1,image2,keypointsImage2,
                              good,None,matchesMask = matchesMask,flags = 2)
cv2.imwrite('Results/task1_matches.jpg',inlierImage)

# Getting corners of image 1 in the 2nd plane
h, w, d = image1.shape
image1Corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
image1CornersPlane2 = np.squeeze(cv2.perspectiveTransform(image1Corners,H))

xMin, yMin, xMax, yMax = getExtremePoints(image1CornersPlane2)

t1 = (xMax-xMin, yMax-yMin)
t2 = (len(image2[0])-int(xMin), len(image2)-int(yMin))
finalImageShape = max(t1,t2)


if xMin < 0 and yMin < 0:
    translate = np.float32([[1,0, -xMin], [0,1, -yMin], [0,0,1]])
elif xMin < 0:
    translate = np.float32([[1,0, -xMin], [0,1,0], [0,0,1]])
elif xMin < 0:
    translate = np.float32([[1,0,0], [0,1, -yMin], [0,0,1]])
else:
    translate = np.float32([[1,0,0], [0,1,0], [0,0,1]])

# Applying homography to image1 to warp it to image2 
finalImage = cv2.warpPerspective(image1, np.matmul(translate,H), finalImageShape)

# Slicing the image with warped image1 to place image2 
finalImage[-int(yMin):-int(yMin)+len(image2), -int(xMin):-int(xMin)+len(image2[0])]=image2
cv2.imwrite('Results/task1_pano.jpg',finalImage)

