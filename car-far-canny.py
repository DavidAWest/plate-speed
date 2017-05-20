#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import pyANPD

import numpy as np

cap = cv2.VideoCapture("C:/DEV/gopro/15slow.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if __name__ == '__main__':
    def nothing(*arg):
        pass
    img = cv2.imread('car-far.png')  # read the image

    height,width = img.shape[:2]

    cv2.namedWindow('edge')
    cv2.createTrackbar('frame', 'edge', 1, total_frames, nothing)
    cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)
    cv2.createTrackbar('kernel_scale', 'edge', 1, 25, nothing)


    while True:
        frame = cv2.getTrackbarPos('frame', 'edge')
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        kernel_scale = cv2.getTrackbarPos('kernel_scale', 'edge')

        cap.set(1, frame);
        ret, frame = cap.read()

        frame = frame[0.25*height:0.6*height, 0.3*width:0.7*width]
        frame = pyANPD.process_image(frame, 0,kernel_scale, thrs1, thrs2, type='est')

        cv2.imshow('edge', frame)
        ch = cv2.waitKey(5)
        if ch == 27:
            break
cv2.destroyAllWindows()