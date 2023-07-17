#!/usr/bin/env python

import cv2
import sys

cam = cv2.VideoCapture(0)
while True:
    ret,frame = cam.read()
    cv2.imshow("test", frame)
    res = cv2.waitKey(1)
    if res is None or res == -1:
        continue
    print('You pressed %d (0x%x), LSB: %d (%s)' % (res, res, res % 256,
        repr(chr(res%256)) if res%256 < 128 else '?'))

