#from https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images

import cv2
import numpy as np

img1 = cv2.imread("/home/rtlabor/Schreibtisch/ir-racecar/OpenCVTests/imgOn.png")
img2 = cv2.imread("/home/rtlabor/Schreibtisch/ir-racecar/OpenCVTests/imgOff.png")
diff = cv2.absdiff(img1, img2)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)


for th in range(255):
    imask =  mask>th

    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]

    cv2.imwrite("/home/rtlabor/Schreibtisch/ir-racecar/OpenCVTests/result{}.png".format(th), canvas)

