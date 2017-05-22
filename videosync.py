import cv2
import numpy as np

import pyANPD
from tools import Buffer, plotline, predict



NONE_ARRAY = np.array(None)

PLOT_WIN = 'plot'
PLOT_WIN_HEIGHT = 300
PLOT_WIN_WIDTH = 800

PLOT_PTS = 60
PLOT_MAX = 49  # max km/h to show
x_scale = 20
y_scale = 6

DIFF_OFFSET = y_scale * PLOT_MAX / 2
DIFF_Y_SCALE = 0.25

font = cv2.FONT_HERSHEY_SIMPLEX

vid_in = cv2.VideoCapture("C:/DEV/gopro/15slow.mp4")

width = int(vid_in.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cropping bounds:
Y_UPPER = int(0.7 * width)
Y_LOWER = int(0.3 * width)
X_UPPER = int(0.6 * height)
X_LOWER = int(0.25 * height)

vid_fps = vid_in.get(cv2.CAP_PROP_FPS)
ms_per_frame = 1000 / vid_fps

write_lin_regfile = False
show_vid = True
LIN_REG_FILENAME = "regression/15_lin_reg_div.csv"

DIFF_DEQUE_SIZE = 60
diffs_buffer = Buffer(DIFF_DEQUE_SIZE, type=np.float)
sizes_buffer = Buffer(120)
predictions_buffer = Buffer(120, type=np.float)
last_plates = pyANPD.PlateBuffer(15)

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
        self.plt_data = Buffer(PLOT_PTS)
        self.plt_color = plot_color

        self.next_record = self.lines[1]
        self.next_gps_time = self.next_record[0] + 1000

    def process_frame(self, capture):
        if abs(capture.get(cv2.CAP_PROP_POS_MSEC) - self.next_gps_time) <= ms_per_frame:
            self.speed = self.next_record[1]
            self.plt_data.put(int(self.speed))
            self.i += 1
            self.next_record = self.lines[self.i]
            self.next_gps_time = self.next_record[0] + 1000

    # draw last n data points as a line
    def draw_plot_line(self, img):
        plotline(self.plt_data.list, img, x_scale, y_scale, PLOT_MAX*y_scale, self.plt_color)

car1 = Car(csv_file="gps_interpolated/15_intp_car1.csv", plot_color=(0, 0, 255))
car2 = Car(csv_file="gps_interpolated/15_intp_car2.csv", plot_color=(255, 0, 0))

# Select a single plate with largest size
def find_plate(img):
    #plates = pyANPD.find_contours(img, 0, 15, 2782, 2930, type='est')
    plates = pyANPD.find_contours(img, 0, 15, 1661, 2784, type='est')
    if len(plates) < 1:
        return None
    # Filter plates which have a distance larger than X to the median of previous plates
    prev_plates = last_plates.list[last_plates.list != NONE_ARRAY]
    if len(prev_plates) > 0:
        center_x = int(np.median(map(lambda p: p.center[0], prev_plates)))
        center_y = int(np.median(map(lambda p: p.center[1], prev_plates)))
        radius = 150
        cv2.circle(img, (center_x, center_y),  radius, (0,0,255))
        plates = filter(lambda p: p.distance_to((center_x, center_y)) < radius, plates)

    if len(plates) is 1:
        return plates[0]
    elif len(plates) > 1:
        # Choose largest
        max = lambda a, b: a if (a.size > b.size) else b
        return reduce(max, plates)
    else: return None;




if write_lin_regfile:
    out = open(LIN_REG_FILENAME, 'w')


def update_plate_buffers(plate):
    if plate is None:
        last_plates.append(last_plates[-1])
        sizes_buffer.repeat_val()
        diffs_buffer.repeat_val()
    else:
        last_plates.append(plate)
        sizes_buffer.put(np.prod(plate.size))

        a = np.mean(sizes_buffer[:60])
        b = np.mean(sizes_buffer[60:])
        diff = b / a
        diffs_buffer.put(diff)


def draw_stats_on_frame():
    if plate is not None:
        # render plate contour and size
        med_size = int(np.median(sizes_buffer.list))
        cv2.drawContours(frame, [plate.box], 0, (127, 0, 255), 2)
        cv2.putText(frame, str(med_size), (plate.x1, plate.y2), font, 1, (255, 0, 0))

    # black background for plot
    cv2.rectangle(plots_speeds, (0, 0), (PLOT_PTS * x_scale, PLOT_MAX * y_scale), (0, 0, 0))

    # base (middle) line for plate diff
    cv2.line(plots_diffs, (0, DIFF_OFFSET), (x_scale * DIFF_DEQUE_SIZE, DIFF_OFFSET), (0, 0, 0))
    plotline(diffs_buffer.list, plots_diffs, x_scale, DIFF_Y_SCALE, DIFF_OFFSET, (0, 255, 0))

    # plot velocity diff
    velo_diffs = car1.plt_data.list - car2.plt_data.list
    plotline(velo_diffs, plots_diffs, x_scale, y_scale, DIFF_OFFSET, (255, 0, 255))
    # plot actual car speeds
    car1.draw_plot_line(plots_speeds)
    car2.draw_plot_line(plots_speeds)
    cv2.putText(plots_speeds, '%.2f' % car1.speed, (PLOT_WIN_WIDTH - 100, PLOT_WIN_HEIGHT - 100), font, 1,
                car1.plt_color)
    cv2.putText(plots_speeds, '%.2f' % car2.speed, (PLOT_WIN_WIDTH - 100, PLOT_WIN_HEIGHT - 50), font, 1,
                car2.plt_color)
    ## plot estimated speed
    plotline(predictions_buffer.list, plots_speeds, x_scale, y_scale, PLOT_MAX * y_scale, (0, 255, 0))
    cv2.putText(plots_speeds, '%.2f' % predictions_buffer[-1], (PLOT_WIN_WIDTH - 100, PLOT_WIN_HEIGHT - 150), font, 1,
                (0, 255, 0))
    cv2.putText(plots_speeds, 'error:' + '%.2f' % (predictions_buffer[-1] - car2.speed),
                (PLOT_WIN_WIDTH - 150, PLOT_WIN_HEIGHT - 200), font,
                1,
                (0, 255, 0))


while vid_in.isOpened():
    ret, frame = vid_in.read()
    if not ret: break;

    millis = vid_in.get(cv2.CAP_PROP_POS_MSEC)
    if millis / 1000 % 10 < 0.05: print millis / 1000; # print progress update

    # try to update car's speed attribute for this frame
    car1.process_frame(vid_in)
    car2.process_frame(vid_in)

    #crop frame, resize
    # frame = cv2.resize(frame, (768, 432))
    frame = frame[X_LOWER:X_UPPER, Y_LOWER:Y_UPPER]

    plots_speeds = np.ones((PLOT_WIN_HEIGHT, PLOT_WIN_WIDTH,3), np.uint8) * 255
    plots_diffs = np.ones((PLOT_WIN_HEIGHT, PLOT_WIN_WIDTH,3), np.uint8) * 255

    plate = find_plate(frame)
    update_plate_buffers(plate)

    # predict car2 speed
    predictions_buffer.put(predict(diffs_buffer[-1], car1.speed))

    if write_lin_regfile:
        out.write(str(diffs_buffer[-1]) + "," + str(car1.speed) + "," + str(car2.speed) + '\n')

    if show_vid:
        draw_stats_on_frame()

        cv2.imshow('main', frame)
        cv2.imshow(PLOT_WIN, plots_speeds)
        cv2.imshow("diffs", plots_diffs)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid_in.release()
cv2.destroyAllWindows()
if write_lin_regfile:
    out.close()
