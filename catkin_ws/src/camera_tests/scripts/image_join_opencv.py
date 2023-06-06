from image_join import ImageJoin
import cv2 as cv


class ImageJoinOpenCV(ImageJoin):
    def __init__(self, ratio = 0.5, min_match=3, smoothing_window_size = 10, stitchter_type = cv.Stitcher_PANORAMA, logger = None) :
        self.ratio=ratio
        self.min_match=min_match
        self.stich = cv.Stitcher.create(stitchter_type)
        self.smoothing_window_size=smoothing_window_size
        super().__init__()


    def blending(self,img1,img2):
        img = []
        img.append(img1)
        img.append(img2)
        result = self.stich.stitch(img)
        if result[0] == cv.Stitcher_OK:
            return result[1]
        else:
            raise Exception("Image stitching failed, is the overlap big enough?")
