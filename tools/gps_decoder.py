#!/usr/bin/python
from __future__ import print_function  # Only needed for Python 2
from optparse import OptionParser
from datetime import datetime

parser = OptionParser()
parser.add_option("-i", "--input", dest="input_filename",
                  help="read input from gps logger trace file")
parser.add_option("-o", "--output", dest="output_filename",
	help="output file to write offset values to")
parser.add_option("-f", "--offset", type="int", dest="offset_frames",
                  help="frame of the video at which this gps data starts")

(options, args) = parser.parse_args()

epoch = datetime.utcfromtimestamp(0)

ms_per_frame = 33.3666666667

# car 1
# offset_frames = 301
# input_filename = "gps/15 slow drive car1.txt"
# output_filename = 'gps2/15_mod_car1.csv'
offset_frames = options.offset_frames
input_filename = options.input_filename
output_filename = options.output_filename

# car 2
# offset_frames = 101
# input_filename = "gps/15 slow drive car2.txt"
# output_filename = 'gps2/15_mod_car2.csv'

offset_ms = offset_frames * ms_per_frame

with open(input_filename) as f:
    lines = f.readlines()

header = lines[0].split(",")



def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

# Take the first timestamp as the min to subtract from all others
t0 = unix_time_millis(datetime.strptime(lines[1].split(",")[0], "%Y-%m-%dT%H:%M:%S.%fZ"))

out = open(output_filename, 'w')

# write header
out.write(lines[0])

for line in lines[1:]:  # Skip the header line
    line_cols = line.split(",")

    # change time to ms timestamp that takes into account video offset
    timestamp = unix_time_millis(datetime.strptime(line_cols[0], "%Y-%m-%dT%H:%M:%S.%fZ"))
    timestamp -= t0
    line_cols[0] = str(timestamp + offset_ms)

    # convert m/s to km/h
    speed = line_cols[6]
    if speed == '':
        speed = 0
    line_cols[6] = str(float(speed) * 3.6)

    # write line
    out.write(",".join(line_cols))

out.close()
f.close()



