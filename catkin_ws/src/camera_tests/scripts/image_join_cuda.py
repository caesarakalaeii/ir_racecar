'''
Modified version of https://github.com/linrl3/Image-Stitching-OpenCV
Improved runtime and more options thanks to static matrix option.
Recommended for most uses.
Homography uses RANSAC
Default for finding features is SIFT
TODO: Time transformation to gpuMat, and transformation it self
'''


import time
from image_join import ImageJoin
from logger import Logger
import cv2 as cv
import numpy as np


class ImageJoinCuda(ImageJoin):

    def __init__(self, ratio=0.85, min_match=10, smoothing_window_size=50, matching_write = False, static_matrix = True, static_mask = False , logger = None, finder = None, matcher = None) :
        self.ratio=ratio
        self.min_match=min_match
        self.matcher = matcher
        self.logger = logger
        if logger == None:
            self.logger = Logger(True, True)
        if matcher == None:
            self.matcher = cv.BFMatcher_create()
        if finder == None:
            try:
                self.finder=cv.AKAZE_create() #maybe replace with ORB or AKAZE
                
            except AttributeError:
                #for older versions of open cv
                try:
                    self.finder=cv.xfeatures2d.SIFT_create()
                except AttributeError:
                    self.l.fail("Unsupported CV version, exiting")
                    exit(1)
        else:
            self.finder = finder
        self.smoothing_window_size=smoothing_window_size
        self.matching_write = matching_write
        self.static_matrix = static_matrix
        self.matrix_set = False
        self.static_mask = static_mask
        self.mask_set = False
        self.mask1 = None
        self.mask2 = None
        self.is_cuda = False
        

        super().__init__()

    
        
        

    def registration(self,img1,img2):
        kp1, des1 = self.finder.detectAndCompute(img1, None)
        kp2, des2 = self.finder.detectAndCompute(img2, None)
        
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        
        
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv.findHomography(image2_kp, image1_kp, cv.RANSAC,5.0)
            self.matrix_set = True
            if(self.static_matrix):
                self.H = H
                
            return H

    def create_mask(self,img1,img2,version, hasDepth = True):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        if not hasDepth:
            return cv.merge([mask])

        return cv.merge([mask, mask, mask])

    def blending(self,img1,img2):
        
        if(self.static_matrix and self.matrix_set):
            return self.blending_no_reg(img1, img2, self.H)
        t = time.time()
        self.H = self.registration(img1,img2)
        self.logger.info(f"Time for registration: {time.time()-t}")
        return self.blending_no_reg(img1, img2, self.H)
        
    
    def blending_no_reg(self,img1,img2, H):
        total_start  = time.time()
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        depth = 0
        try:
            depth = img1.shape[2]
        except:
            depth = 0
        if depth == 0:
            panorama1 = np.zeros((height_panorama, width_panorama))
            if self.static_mask and self.mask_set:
                mask1 = self.mask1
            else:
                mask_time = time.time()
                mask1 = self.create_mask(img1,img2,version='left_image', hasDepth=False)
                self.logger.info(f"Time for masking: {time.time()-mask_time}")
                self.mask1 =  mask1
            panorama1[0:img1.shape[0], 0:img1.shape[1]] = img1
            panorama1 = panorama1*mask1 #evtl durch primitive ersetzen (a*b)
            if self.static_mask and self.mask_set:
                mask2 = self.mask2
            else:
                mask_time = time.time()
                mask2 = self.create_mask(img1,img2,version='right_image', hasDepth=False)
                self.mask2 = mask2
                self.logger.info(f"Time for masking: {time.time()-mask_time}")
                self.mask_set = True
            try:
                start = time.time()
                src = cv.cuda.GpuMat(img2)
                convertGPU = time.time()
                M = cv.UMat(H)
                convertUMat = time.time()
                warped = cv.cuda.warpPerspective(src, M, dsize = (width_panorama, height_panorama)).download()
                warponGPU = time.time()
                panorama2 = warped * mask2
                end = time.time()
                self.logger.info(f"Time to transform to GPUMat: {convertGPU-start}\nTime to transform to UMat: {convertGPU-convertUMat}\nTime to warp on GPU: {convertUMat-warponGPU}\nTotal elapsed time: {end-start}\n")
            except:
                raise Exception("Couldn't match images.")
            start = time.time()
            result=panorama1+panorama2 #evtl durch primitive ersetzen (a+b)
            end = time.time()
            self.logger.info(f"Time to add images on CPU: {end-start}")

            rows, cols = np.where(result[:, :] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col]
            a = time.time()
            self.logger.info(f"Time for NP stuff: {a-end}")

        else :
            panorama1 = np.zeros((height_panorama, width_panorama, depth))
            mask1 = self.create_mask(img1,img2,version='left_image')
            panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
            panorama1 = panorama1*mask1
            mask2 = self.create_mask(img1,img2,version='right_image')
            start = time.time()
            src = cv.cuda.GpuMat(img2)
            convertGPU = time.time()
            M = cv.UMat(H)
            convertUMat = time.time()
            warped = cv.cuda.warpPerspective(src, M, dsize = (width_panorama, height_panorama)).download()
            warponGPU = time.time()
            panorama2 = warped * mask2
            end = time.time()
            self.logger.info(f"Time to transform to GPUMat: {convertGPU-start}\nTime to transform to UMat: {convertUMat-convertGPU}\nTime to warp on GPU: {convertUMat-warponGPU}\nTotal elapsed time: {end-start}\n")
            start = time.time()
            result=panorama1+panorama2
            end = time.time()
            self.logger.info(f"Time to add images on GPU: {end-start}")
            rows, cols = np.where(result[:, :, 0] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col, :]
            a = time.time()
            self.logger.info(f"Time for NP stuff: {a-end}")
        total_end  = time.time()
        self.logger.info(f"Total time to join: {total_end-total_start}")
        return cv.convertScaleAbs(final_result)
