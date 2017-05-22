#!/usr/bin/env python

from __future__ import print_function  # Python 2/3 compatibility

import cv2
import pyANPD

SLIDER_FRAME = 'frame'
SLIDER_KERNEL_SCALE = 'kernel_scale'
SLIDER_THRESH2 = 'threshold1'
SLIDER_THRESH1 = 'threshold2'
WIN_NAME = 'main_window'

cap = cv2.VideoCapture("C:/DEV/gopro/14 slow.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def plate_finder(_=None, paused= True):
    global cap, cfg

    setattr(cfg, SLIDER_FRAME, cv2.getTrackbarPos(SLIDER_FRAME, WIN_NAME))
    setattr(cfg, SLIDER_THRESH1, cv2.getTrackbarPos(SLIDER_THRESH1, WIN_NAME))
    setattr(cfg, SLIDER_THRESH2, cv2.getTrackbarPos(SLIDER_THRESH2, WIN_NAME))
    setattr(cfg, SLIDER_KERNEL_SCALE, cv2.getTrackbarPos(SLIDER_KERNEL_SCALE, WIN_NAME))

    if paused: cap.set(1, cfg.frame);  # set which frame to work with
    ret, frame = cap.read()

    frame = frame[int(0.25 * height):int(0.6 * height), int(0.3 * width):int(0.7 * width)]
    plates = pyANPD.find_contours(frame, 0, cfg.kernel_scale, cfg.threshold1, cfg.threshold2, type='est')
    for plate in plates:
        cv2.drawContours(frame, [plate.box], 0, (127, 0, 255), 2)

    cv2.imshow(WIN_NAME, frame)


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
    cv2.createTrackbar(SLIDER_THRESH1, WIN_NAME, cfg.threshold1, 7000, plate_finder)
    cv2.createTrackbar(SLIDER_THRESH2, WIN_NAME, cfg.threshold2, 7000, plate_finder)
    cv2.createTrackbar(SLIDER_KERNEL_SCALE, WIN_NAME, cfg.kernel_scale, 30, plate_finder)


    # run plate_finder() with pausing/playing with the Space key:
    while cap.isOpened():
        plate_finder()
        key = cv2.waitKey(0)
        if key == 32:
            while cap.isOpened():
                plate_finder(paused=False)
                key = cv2.waitKey(1)
                if key == 32:
                    cfg.frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.setTrackbarPos(SLIDER_FRAME, WIN_NAME, cfg.frame)
                    break
    cv2.destroyAllWindows()