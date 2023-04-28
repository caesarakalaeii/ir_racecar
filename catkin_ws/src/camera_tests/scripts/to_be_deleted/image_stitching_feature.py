import cv2
import numpy as np
import time as t
#CUDA unlikely to help, because it would eat up all resources and not allow other compute invensive loads to funktion in real time!
#Not really a approach that will work in real time on this hardware

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.SIFT_create() #maybe replace wir ORB or AKAZE
        self.smoothing_window_size=100
        self.matching_write = False


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
            return H

    def create_mask(self,img1,img2,version):
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
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        return self.blending_no_reg(img1, img2, H)
        
    
    def blending_no_reg(self,img1,img2, H):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        start_mask = t.time()
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        
        start_warp = t.time()
        print("time to mask: ", start_warp-start_mask)
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        end_warp = t.time()
        print("Time to warp: ", end_warp-start_warp)
        result=panorama1+panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = np.min(rows), np.max(rows) + 1
        min_col, max_col = np.min(cols), np.max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        rest = t.time()
        print("Time for rest: ", rest-end_warp)
        cv2.imshow("Test", final_result)
        return cv2.convertScaleAbs(final_result)
    
def test(argv1,argv2, loop):
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    total = 0
    blending_reg = 0
    calcH_time = 0
    blending_no_reg = 0
    for i in range(loop):
        ret, img1 = cam1.read()
        ret, img2 = cam2.read()
        start = t.time()
        #final=Image_Stitching().blending(img1,img2)
        first = t.time()
        H = Image_Stitching().registration(img1, img2)
        calcH = t.time()
        final = Image_Stitching().blending_no_reg(img1, img2, H)
        cv2.imshow("Test", final)
        end = t.time()
        total += end-start
        blending_reg += first-start 
        calcH_time += calcH-first
        blending_no_reg += end-calcH

    print("Total time elapsed: ", total*1000, "ms")
    print("Time for blending with reg: ", blending_reg*1000, "ms")
    print("Time for calculating H: ", calcH_time*1000, "ms")
    print("Time for Blending with given H: ", blending_no_reg*1000, "ms")
    print("Average time elapsed: ", total/loop*1000, "ms")
    print("Average time for blending with reg: ", blending_reg/loop*1000, "ms")
    print("Average time for calculating H: ", calcH_time/loop*1000, "ms")
    print("Average time for Blending with given H: ", blending_no_reg/loop*1000, "ms")
    print("Time to achieve real time capabilities: ", 1000/30 , "ms")
    print("Difference to real time:", 1000/30 - blending_no_reg/loop*1000, "ms")

def main(argv1,argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final=Image_Stitching().blending(img1,img2)
    cv2.imwrite('panorama.jpg', final)
    



if __name__ == '__main__':
    path1= "/home/rtlabor/Bilder/Kamera/image21.jpg"
    path2= "/home/rtlabor/Bilder/Kamera/image22.jpg"
    try: 
        test(path1, path2, 10)
    except Exception:
        print("Somethings Wrong")
