import collections
import cv2
import pyANPD


PLOT_PTS_MINUS_ONE = 9
PLOT_MAX = 30  # max km/h to show
x_scale = 20
y_scale = 4

font = cv2.FONT_HERSHEY_SIMPLEX

vid_in = cv2.VideoCapture("C:/DEV/gopro/15slow.mp4")
vid_fps = vid_in.get(cv2.CAP_PROP_FPS)
ms_per_frame = 1000 / vid_fps


class Car():
    """ Car-related GPS data.
        Reads csv with gps data into memory for use with video """

    def __init__(self, csv_file, plot_color):
        with open(csv_file) as f:
            self.lines = f.readlines()
        f.close()

        # pre-parse the data a bit, convert strings to floats
        for j in range(1, len(self.lines)):  # skip header row
            line = self.lines[j].split(",")[:7]
            line[0] = float(line[0])  # convert timestamp string to float
            line[6] = float(line[6])  # convert speed string to float
            self.lines[j] = line

        self.i = 1  # counter indicating which timestamp we'll match to the video next
        self.speed = None

        # plotting
        self.plt_data = collections.deque([0] * (PLOT_PTS_MINUS_ONE+1), maxlen=PLOT_PTS_MINUS_ONE + 1)
        self.plt_color = plot_color

        self.next_gps = self.lines[1]
        self.next_gps_time = self.next_gps[0]

    def process_frame(self, capture):
        if abs(capture.get(cv2.CAP_PROP_POS_MSEC) - self.next_gps_time) <= ms_per_frame:
            self.speed = self.next_gps[6]
            self.plt_data.append(int(self.speed))
            self.i += 1
            self.next_gps = self.lines[self.i]
            self.next_gps_time = self.next_gps[0]

    def draw_plot_line(self, img):
        # draw last 10 data points as a line
        for j in range(PLOT_PTS_MINUS_ONE):
            cv2.line(img,
                     (j * x_scale, (PLOT_MAX - self.plt_data[j]) * y_scale),
                     (j * x_scale + x_scale, (PLOT_MAX - self.plt_data[j + 1]) * y_scale),
                     self.plt_color)


car1 = Car(csv_file="gps2/15_mod_car1.csv", plot_color=(0, 0, 255))
car2 = Car(csv_file="gps2/15_mod_car2.csv", plot_color=(255, 0, 0))

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('output.avi', fourcc, vid_fps, (768, 432))

def nothing(*arg):
	pass

while vid_in.isOpened():
    ret, frame = vid_in.read()

    # does this frame correspond to a cars next gps record?
    car1.process_frame(vid_in)
    car2.process_frame(vid_in)


    frame = cv2.resize(frame, (768, 432))

    #frame = pyANPD.process_image(frame, 0, cv2.getTrackbarPos('thrs1', 'edge'), cv2.getTrackbarPos('thrs2', 'edge'), type='eu')
    frame = pyANPD.process_image(frame, 0, 15, 2782, 2930, type='est')

    cv2.putText(frame, str(car1.speed), (200, 50), font, 1, car1.plt_color)
    cv2.putText(frame, str(car2.speed), (200, 100), font, 1, car2.plt_color)

    # black background for plot
    cv2.rectangle(frame, (0, 0), (PLOT_PTS_MINUS_ONE * x_scale, PLOT_MAX * y_scale), (255, 255, 255), cv2.FILLED)
    car1.draw_plot_line(frame)
    car2.draw_plot_line(frame)

    cv2.imshow('edge', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid_in.release()
# out.release()
cv2.destroyAllWindows()