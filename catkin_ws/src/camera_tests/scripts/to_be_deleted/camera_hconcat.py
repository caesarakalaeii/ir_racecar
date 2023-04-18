import cv2
import numpy as np

class Image_Stitching(object):

    def __init__(self) -> None:
        self.left_y_offset = 20
        self.left_x_offset = 0
        self.right_y_offset = 0
        self.right_x_offset = 0
        pass


    def blending(self, img1, img2):
        shape1 = np.shape(img1)
        shape2 = np.shape(img2)
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
    
def test(argv1,argv2, loop):
    import time as t
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    total = 0
    for _ in range(loop):
        start = t.time()
        Image_Stitching().blending(img1,img2)
        end = t.time()
        total += end-start

    print("Total time elapsed: ", total*1000, "ms")
    print("Average time elapsed: ", total/loop*1000, "ms")
    print("Difference to real time:", 1000/30 - total/loop*1000, "ms")

if __name__ == '__main__':
    path1= "/home/rtlabor/Bilder/Kamera/image21.jpg"
    path2= "/home/rtlabor/Bilder/Kamera/image22.jpg"
    try: 
        test(path1, path2, 100)
    except Exception:
        print("Somethings Wrong")