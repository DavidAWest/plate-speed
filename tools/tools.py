import cv2
import numpy as np


cf_intercept = 21.276770
cf_vel1 = 1.308432
cf_vel1_sqd = -0.013927
cf_diff = -27.641055
cf_diff_sqd =  5.164449

def predict(difference, speed):
    return cf_intercept + \
           cf_vel1*speed + cf_vel1_sqd*np.power(speed,2) + \
           cf_diff*difference + cf_diff_sqd*np.power(difference,2)

class Buffer():
    def __init__(self, size, type=np.int):
        self.list = np.zeros(size, dtype=type)

    def __iter__(self):
        return iter(self.list)

    def __getitem__(self, item):
        return self.list[item]

    # equivalent to appending the current last element to the end of this deque.
    def repeat_val(self):
        self.list = np.roll(self.list, -1)
        self.list[-1] = self.list[-2]

    def put(self, val):
        self.list = np.roll(self.list, -1)
        self.list[-1] = val


def plotline(vals, window, xscale, yscale, y_offset=0, color=(0, 0, 0)):
    xlen = len(vals)
    for j in range(xlen - 1):
        if np.isfinite(vals[j]) and np.isfinite(vals[j+1]):
            cv2.line(window,
                     (j * xscale, int((y_offset - (vals[j] * yscale)))),
                     (j * xscale + xscale, int((y_offset - (vals[j + 1] * yscale)))),
                     color)


NONE_ARRAY = np.array(None)