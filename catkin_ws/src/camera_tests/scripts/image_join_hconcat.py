'''
Simple class, to stitch two Images together, no fluff only cropping and concentation.
Recommended for very light weight applications
'''


from image_join import ImageJoin
import numpy as np
import cv2 as cv

class ImageJoinHConcat(ImageJoin):

    def __init__(self, left_y_offset = 20, right_y_offset=0, left_x_offset=0, right_x_offset=0, logger = None):
        super().__init__(logger)
        self.left_y_offset = left_y_offset
        self.left_x_offset = left_x_offset
        self.right_y_offset = right_y_offset
        self.right_x_offset = right_x_offset


    def blending(self, img1, img2):
        
        shape1 = img1.shape
        shape2 = img2.shape
        if self.left_y_offset != 0: #add black bar on top / bottom of imgages
            y_offset = np.zeros((abs(self.left_y_offset),shape1[1], shape1[2]))
            if self.left_y_offset < 0:
                img1= np.append(img1, y_offset, axis=0) 
                img2= np.append(y_offset, img1, axis=0) #add buffer fo left/right image to preserve shape
            else:
                img1= np.append(y_offset, img1, axis=0)
                img2= np.append(img2, y_offset, axis=0)
        if self.right_y_offset != 0:
            y_offset = np.zeros((abs(self.right_y_offset),shape1[1], shape1[2]))
            if self.right_y_offset < 0:
                img2= np.append(img2, y_offset, axis=1)
                img1= np.append(y_offset, img1, axis=1)
            else:
                img2= np.append(y_offset, img1, axis=1)
                img1= np.append(y_offset, img1, axis=1)
        if self.left_x_offset != 0:         #cut part from left/right
            img1 = img1[:,:-self.left_x_offset,:]
        if self.right_x_offset != 0:
            img2 = img2[:,-self.right_x_offset:,:]
        img = np.append(img1, img2, axis=1) #stack horizontally
        return cv.convertScaleAbs(img)  #convert to bgr8 compatible type and return
