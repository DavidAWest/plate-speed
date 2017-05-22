import cv2
import numpy as np


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
        cv2.line(window,
                 (j * xscale, int((y_offset - (vals[j] * yscale)))),
                 (j * xscale + xscale, int((y_offset - (vals[j + 1] * yscale)))),
                 color)
