import numpy as np
import cv2



class ImageJoin(object):
    

    def __init__(self):
        pass

    def blending(self, img1, img2):
        pass

class ImageJoinFactory():
    default_list = {
        "camera1": "joined_cams/usb_cam1/image_raw",
        "camera2": "joined_cams/usb_cam2/image_raw",
        "publish": "joined_image/image_raw",
        "queue_size": 10,
        "encoding": 'bgr8',
        "verbose": False,
        "joinType": 1,
        "left_y_offset": 20,
        "right_y_offset": 0,
        "left_x_offset": 0,
        "right_x_offset": 0,
        "ratio": 0.85,
        "min_match": 10,
        "smoothing_window_size": 50,
        "matching_write": False,
        "static_matrix": False,
        "static_mask": False,
        "stitchter_type": cv2.Stitcher_PANORAMA,
        "direct_import": False,
        "direct_import_sources": (0,2)
    }
    

    def create_instance(**kwargs):
        for k, v in ImageJoinFactory.default_list:
            if k in kwargs:
                continue
            else: kwargs.update(k=v)
        joinType = kwargs["joinType"]
        if joinType == 1:
            return ImageJoinHConcat(kwargs["left_y_offset"],
                                    kwargs["right_y_offset"],
                                    kwargs["left_x_offset"],
                                    kwargs["right_x_offset"])
        elif joinType == 2:
            return ImageJoinFeature(kwargs["ratio"],
                                    kwargs["min_match"],
                                    kwargs["smoothing_window_size"],
                                    kwargs["matching_write"],
                                    kwargs["static_matrix"],
                                    kwargs["static_mask"])
        elif joinType == 3:
            return ImageJoinOpenCV(kwargs["ratio"],
                                   kwargs["min_match"],
                                   kwargs["smoothing_window_size"],
                                   kwargs["stitchter_type"])
        else:
            raise ValueError("JoinType not known, please use either CONCAT = 1, FEATURE = 2 or OPENCV = 3")

class ImageJoinHConcat(ImageJoin):

    def __init__(self, left_y_offset = 20, right_y_offset=0, left_x_offset=0, right_x_offset=0):
        super().__init__()
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
        return cv2.convertScaleAbs(img)  #convert to bgr8 compatible type and return
    
class ImageJoinFeature(ImageJoin):

    def __init__(self, ratio=0.85, min_match=10, smoothing_window_size=50, matching_write = False, static_matrix = False, static_mask = False ) :
        self.ratio=ratio
        self.min_match=min_match
        self.sift=cv2.SIFT_create() #maybe replace wir ORB or AKAZE
        self.smoothing_window_size=smoothing_window_size
        self.matching_write = matching_write
        self.static_matrix = static_matrix
        self.matrix_set = False
        self.static_mask = static_mask
        self.mask_set = False
        self.mask1 = None
        self.mask2 = None

        super().__init__()


    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        if self.matching_write:
            cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
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
            return cv2.merge([mask])

        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        if(self.static_matrix and self.matrix_set):
            return self.blending_no_reg(img1, img2, self.H)
        self.H = self.registration(img1,img2)
        return self.blending_no_reg(img1, img2, self.H)
        
    
    def blending_no_reg(self,img1,img2, H):
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
                mask1 = self.create_mask(img1,img2,version='left_image', hasDepth=False)
                self.mask1 =  mask1
            panorama1[0:img1.shape[0], 0:img1.shape[1]] = img1
            panorama1 *= mask1
            if self.static_mask and self.mask_set:
                mask2 = self.mask2
            else:
                mask2 = self.create_mask(img1,img2,version='right_image', hasDepth=False)
                self.mask2 = mask2
                self.mask_set = True
            panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
            result=panorama1+panorama2

            rows, cols = np.where(result[:, :] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col]

        else :
            panorama1 = np.zeros((height_panorama, width_panorama, depth))
            mask1 = self.create_mask(img1,img2,version='left_image')
            panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
            panorama1 *= mask1
            mask2 = self.create_mask(img1,img2,version='right_image')
            panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
            result=panorama1+panorama2

            rows, cols = np.where(result[:, :, 0] != 0)
            min_row, max_row = np.min(rows), np.max(rows) + 1
            min_col, max_col = np.min(cols), np.max(cols) + 1
            final_result = result[min_row:max_row, min_col:max_col, :]
        return cv2.convertScaleAbs(final_result)


class ImageJoinOpenCV(ImageJoin):
    def __init__(self, ratio = 0.5, min_match=3, smoothing_window_size = 10, stitchter_type = cv2.Stitcher_PANORAMA) :
        self.ratio=ratio
        self.min_match=min_match
        self.stich = cv2.Stitcher.create(stitchter_type)
        self.smoothing_window_size=smoothing_window_size
        super().__init__()


    def blending(self,img1,img2):
        img = []
        img.append(img1)
        img.append(img2)
        result = self.stich.stitch(img)
        if result[0] == cv2.Stitcher_OK:
            return result[1]
        else:
            raise Exception("Image stitching failed, is the overlap big enough?")