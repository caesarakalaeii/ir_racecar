import cv2
import time as t

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.5
        self.min_match=3
        self.stich = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        self.smoothing_window_size=10


    def registration(self,img1,img2):
        #maybe?
        return


    def blending(self,img1,img2):
        img = []
        img.append(img1)
        img.append(img2)
        result = self.stich.stitch(img)
        if result[0] == cv2.Stitcher_OK:
            return result[1]
        else:
            raise Exception("Image stitching failed, is the overlap big enough?")
        


def main(argv1,argv2):
    
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    stitcher = Image_Stitching()
    final=stitcher.blending(img1,img2)
    cv2.imwrite('panorama.jpg', final)

def test(argv1,argv2, loop):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    total = 0

    stitcher = Image_Stitching()
    for i in range(loop):
        start = t.time()
        stitcher.blending(img1,img2)
        end = t.time()
        total += end-start

    print("Total time elapsed: ", total, "s")
    print("Average time elapsed: ", total/loop, "s")

if __name__ == '__main__':
    path1= "/home/rtlabor/Bilder/Kamera/image21.jpg"
    path2= "/home/rtlabor/Bilder/Kamera/image22.jpg"
    try: 
        test(path1, path2, 100)
    except Exception:
        print("Somethings Wrong")
