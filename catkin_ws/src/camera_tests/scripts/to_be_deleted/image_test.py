# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:54:26 2023

@author: Caesar
"""

import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find the key points and descriptors using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match the descriptors from both images using a k-nearest neighbors algorithm
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter out bad matches
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

# Get the coordinates of the matched points in both images
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# Find the translation vector using the closest point algorithm
M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

# Merge the two images using the translation vector
merged_img = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))

# Blend the overlapping region in the merged image
mask = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
mask[:,:] = 255
mask = cv2.warpAffine(mask, M, (img2.shape[1], img2.shape[0]))
merged_img = cv2.addWeighted(img2, 0.5, merged_img, 0.5, 0, dtype=cv2.CV_8U)

# Apply the mask to the overlapping region
merged_img[mask != 0] = img2[mask != 0]

# Display the merged image
cv2.imshow('Merged Image', merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
