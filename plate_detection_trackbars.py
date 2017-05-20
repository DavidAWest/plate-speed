#!/usr/bin/env python

from __future__ import print_function  # Python 2/3 compatibility

import cv2
import pyANPD

SLIDER_FRAME = 'frame'
SLIDER_KERNEL_SCALE = 'kernel_scale'
SLIDER_THRESH2 = 'threshold1'
SLIDER_THRESH1 = 'threshold2'
WIN_NAME = 'main_window'

cap = cv2.VideoCapture("C:/DEV/gopro/15slow.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def plate_finder(_=None):
    global cap, cfg
    for slider in [SLIDER_FRAME, SLIDER_THRESH1, SLIDER_THRESH2, SLIDER_KERNEL_SCALE]:
        val = cv2.getTrackbarPos(slider, WIN_NAME)
        setattr(cfg, slider, val)

    cap.set(1, cfg.frame)  # set which frame to work with
    ret, frame = cap.read()

    frame = frame[0.25 * height:0.6 * height, 0.3 * width:0.7 * width]
    frame = pyANPD.process_image(frame, 0, cfg.kernel_scale, cfg.threshold1, cfg.threshold2, type='est')

    cv2.imshow(WIN_NAME, frame)
    print ("updating frame with cfg", cfg)


class PlateDetectionConfig:
    """ Set of params used to find the licenseplate rectangle from a frame"""

    def __init__(self, frame, threshold1, threshold2, kernel_scale):
        self.frame = frame
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.kernel_scale = kernel_scale


if __name__ == '__main__':
    global cfg
    cfg = PlateDetectionConfig(frame=1, threshold1=2782, threshold2=2930, kernel_scale=15)

    cv2.namedWindow(WIN_NAME)
    cv2.createTrackbar(SLIDER_FRAME, WIN_NAME, cfg.frame, total_frames, plate_finder)
    cv2.createTrackbar(SLIDER_THRESH1, WIN_NAME, cfg.threshold1, 5000, plate_finder)
    cv2.createTrackbar(SLIDER_THRESH2, WIN_NAME, cfg.threshold2, 5000, plate_finder)
    cv2.createTrackbar(SLIDER_KERNEL_SCALE, WIN_NAME, cfg.kernel_scale, 30, plate_finder)

    plate_finder()

    cv2.waitKey(0)
    cv2.destroyAllWindows()