import collections
import cv2
import pyANPD
import numpy as np

PLOT_PTS_MINUS_ONE = 29
PLOT_MAX = 30  # max km/h to show
x_scale = 10
y_scale = 4

font = cv2.FONT_HERSHEY_SIMPLEX

vid_in = cv2.VideoCapture("C:/DEV/gopro/15slow.mp4")
vid_fps = vid_in.get(cv2.CAP_PROP_FPS)
ms_per_frame = 1000 / vid_fps

write_lin_regfile = True

class Deque():
    def __init__(self, size):
        self.list = np.zeros(size, dtype=np.int)


    def __iter__(self):
        return iter(self.list)

    def put(self, val):
        self.list = np.roll(self.list, -1)
        self.list[-1] = val


class Car():
    """ Car-related GPS data.
        Reads csv with gps data into memory for use with video """

    def __init__(self, csv_file, plot_color):
        with open(csv_file) as f:
            self.lines = f.readlines()
        f.close()

        # pre-parse the data a bit, convert strings to floats
        for j in range(len(self.lines)):
            line = self.lines[j].split(",")
            line[0] = float(line[0])
            line[1] = float(line[1])
            self.lines[j] = line

        self.i = 1  # counter indicating which timestamp we'll match to the video next
        self.speed = 0.0

        # plotting
        self.plt_data = collections.deque([0] * (PLOT_PTS_MINUS_ONE + 1), maxlen=PLOT_PTS_MINUS_ONE + 1)
        self.plt_color = plot_color

        self.next_record = self.lines[1]
        self.next_gps_time = self.next_record[0]

    def process_frame(self, capture):
        if abs(capture.get(cv2.CAP_PROP_POS_MSEC) - self.next_gps_time) <= ms_per_frame:
            self.speed = self.next_record[1]
            self.plt_data.append(int(self.speed))
            self.i += 1
            self.next_record = self.lines[self.i]
            self.next_gps_time = self.next_record[0]

    def draw_plot_line(self, img):
        # draw last n data points as a line
        for j in range(PLOT_PTS_MINUS_ONE):
            cv2.line(img,
                     (j * x_scale, (PLOT_MAX - self.plt_data[j]) * y_scale),
                     (j * x_scale + x_scale, (PLOT_MAX - self.plt_data[j + 1]) * y_scale),
                     self.plt_color)


car1 = Car(csv_file="gps_interpolated/15_intp_car1.csv", plot_color=(0, 0, 255))
car2 = Car(csv_file="gps_interpolated/15_intp_car2.csv", plot_color=(255, 0, 0))


def find_plate(img):
    plates = pyANPD.find_contours(img, 0, 15, 2782, 2930, type='est')
    # Select a single plate.
    if len(plates) > 0:
        largest_plate = plates[0]
        for plate in plates:
            if plate.size > largest_plate.size:
                largest_plate = plate
        return largest_plate
    else:
        return None

size_deque = Deque(5)

if write_lin_regfile:
    out = open("regression/lin_reg.csv", 'w')


while vid_in.isOpened():
    ret, frame = vid_in.read()

    # print progress update
    millis = vid_in.get(cv2.CAP_PROP_POS_MSEC)
    if millis / 1000 % 10 < 0.1:
        print millis / 1000


    # try to update car's speed attribute for this frame
    car1.process_frame(vid_in)
    car2.process_frame(vid_in)

    frame = cv2.resize(frame, (768, 432))

    plate = find_plate(frame)

    if plate is not None:
        size_deque.put(np.prod(plate.size))
        med_size = int(np.median(size_deque.list))

        diff = size_deque.list[-1]-size_deque.list[0]


        cv2.drawContours(frame, [plate.box], 0, (127, 0, 255), 2)
        cv2.putText(frame, str(med_size), (plate.x1, plate.y2), font, 1, (255, 0, 0))

    if write_lin_regfile:
        out.write( str(diff) + "," + str(car1.speed)+ "," + str(car2.speed) + '\n')



    # black background for plot
    cv2.rectangle(frame, (0, 0), (PLOT_PTS_MINUS_ONE * x_scale, PLOT_MAX * y_scale), (255, 255, 255), cv2.FILLED)
    car1.draw_plot_line(frame)
    car2.draw_plot_line(frame)

    cv2.putText(frame, '%.2f' % car1.speed, (200, 50), font, 1, car1.plt_color)
    cv2.putText(frame, '%.2f' % car2.speed, (200, 100), font, 1, car2.plt_color)

    cv2.imshow('main', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_in.release()
cv2.destroyAllWindows()
if write_lin_regfile:
    out.close()
