import math

import cv2
import numpy as np
import time


def validate_contour(rect, img, aspect_ratio_range, area_range):
    # rect = cv2.minAreaRect(contour)
    img_width = img.shape[1]
    # img_height = img.shape[0]
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # X = rect[0][0]
    # Y = rect[0][1]
    # angle = rect[2]
    width = rect[1][0]
    height = rect[1][1]

    # angle = (angle + 180) if width < height else (angle + 90)

    output = False

    if (width > 0 and height > 0) and ((width < img_width / 2.0) and (height < img_width / 2.0)):
        aspect_ratio = float(width) / height if width > height else float(height) / width
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            if area_range[0] <= width*height <= area_range[1]:

                box_copy = np.copy(box[1:])
                point = np.copy(box[0])

                dists = np.linalg.norm(point-box_copy, axis=1) # finds normalized euclidean distances between the pts
                sorted_dists = np.sort(dists)
                opposite_pt_index = np.where(dists == sorted_dists[1])
                opposite_point = box_copy[opposite_pt_index][0]


                tmp_angle = 90

                if abs(point[0] - opposite_point[0]) > 0:
                    tmp_angle = abs(float(point[1] - opposite_point[1])) / abs(point[0] - opposite_point[0])
                    tmp_angle = rad_to_deg(math.atan(tmp_angle))

                if tmp_angle <= 9:
                    output = True
    return output

class PlateBuffer():
    def __init__(self, size):
        self.list = np.empty(shape=(size,), dtype=object)
    def __iter__(self):
        return iter(self.list)
    def __getitem__(self, item):
        return self.list[item]
    # equivalent to appending the current last element to the end of this deque.
    def put_no_val(self):
        self.list = np.roll(self.list, -1)
        self.list[-1] = self.list[-2]
    def append(self, val):
        self.list = np.roll(self.list, -1)
        self.list[-1] = val

class Plate():
    """ Rectangle of the plate """

    def __init__(self, contour):
        self.contour = contour

        self.rect = cv2.minAreaRect(contour)
        self.box = cv2.boxPoints(self.rect)
        self.box = np.int0(self.box)
        Xs = self.box[:,0]
        Ys = self.box[:,1]

        self.x1 = np.min(Xs)
        self.x2 = np.max(Xs)
        self.y1 = np.min(Ys)
        self.y2 = np.max(Ys)

        self.center = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
        self.size = self.x2 - self.x1, self.y2 - self.y1

    def distance_to(self, point):
        distance = np.linalg.norm( np.subtract(self.center,point)) # Euclidean
        return distance


def deg_to_rad(angle):
    return angle * np.pi / 180.0


def rad_to_deg(angle):
    return angle * 180 / np.pi


def enhance(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]])
    return cv2.filter2D(img, -1, kernel)


def find_contours(raw_image, debug, kernel_scale, thrs1, thrs2, **options):
    se_shape = (16, 4)

    if options.get('type') == 'rect':
        se_shape = (17, 4)

    if options.get('type') == 'est':
        se_shape = (2 * kernel_scale, 1 * kernel_scale)

    elif options.get('type') == 'square':
        se_shape = (7, 6)

    # raw_image = cv2.imread(name,1)  # Jakob's note: this method used to eat filenames
    input_image = np.copy(raw_image)

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    #gray = enhance(gray)
    #cv2.imshow('enhance', gray)

    # gray = cv2.GaussianBlur(gray, (5,5), 0)
    # cv2.imshow('blur', gray)
    #gray = cv2.Sobel(gray, -1, 1, 0)

    canny = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
    cv2.imshow('canny', canny)

    #h,sobel = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, se_shape)
    gray = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, se)

    cv2.imshow('morphologyEx', gray)
    ed_img = np.copy(gray)

    _, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    font = cv2.FONT_HERSHEY_SIMPLEX

    returned_plates = []
    for contour in contours:
        aspect_ratio_range = (2.2, 12)
        area_range = (500, 18000)

        if options.get('type') == 'rect':
            aspect_ratio_range = (2.2, 12)
            area_range = (500, 18000)

        elif options.get('type') == 'square':
            aspect_ratio_range = (1, 2)
            area_range = (300, 8000)

        ##Exact aspect ratio of eu plate is 4.6 (based on 520 x 113 mm)
        elif options.get('type') == 'est':
            aspect_ratio_range = (4, 5)
            area_range = (40, 18000)

        plate = Plate(contour)
        if validate_contour(plate.rect, gray, aspect_ratio_range, area_range):

            angle = plate.rect[2]
            if angle < -45:
                angle += 90

            W = plate.rect[1][0]
            H = plate.rect[1][1]
            # aspect_ratio = float(W) / H if W > H else float(H) / W

            # center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # size = (x2 - x1, y2 - y1)
            M = cv2.getRotationMatrix2D((plate.size[0] / 2, plate.size[1] / 2), angle, 1.0);
            tmp = cv2.getRectSubPix(ed_img, plate.size, plate.center)
            tmp = cv2.warpAffine(tmp, M, plate.size)
            TmpW = H if H > W else W
            TmpH = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(TmpW), int(TmpH)), (plate.size[0] / 2, plate.size[1] / 2))
            __, tmp = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            white_pixels = np.count_nonzero(tmp)
            edge_density = float(white_pixels) / (tmp.shape[0] * tmp.shape[1])

            if edge_density > 0.9:
                returned_plates.append(plate)

    return np.array(returned_plates)