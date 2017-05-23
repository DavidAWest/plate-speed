import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input_filename",
                  help="read input from gps logger trace file")
parser.add_option("-o", "--output", dest="output_filename",
	help="output file to write offset values to")
parser.add_option("-v", "--visualize", action="store_true", dest="visualize", default=False,
                  help="show matplotlib of data")

(options, args) = parser.parse_args()


def data_from_csv(file):
    with open(file) as f:
        lines = np.array(f.readlines())
    f.close()

    data = np.ndarray((len(lines), 8), dtype=float)

    it = np.nditer(lines[1:], flags=['f_index'])
    while not it.finished:
        data[it.index] = np.fromstring(it[0], sep=",", count=8)
        it.iternext()
    times = data[:-1, 0]
    speeds = data[:-1, 6]
    return times, speeds


input_filename = options.input_filename
output_filename = options.output_filename



times, speeds = data_from_csv(input_filename)
if len(speeds) != len(times):
    print "Speeds array not same length as times array!"
    exit()

xnew = np.arange(times[0], times[-1], 33.3666666667)
ynew = np.interp(xnew, times, speeds)


if len(xnew) != len(ynew):
    print "Array length mismatch, theres a bug in the code!"
    exit()

out = open(output_filename, 'w')
for i in range(len(xnew)):
    out.write( str(xnew[i]) + "," + str(ynew[i]) + '\n')
out.close()

if options.visualize:
	plt.plot(times, speeds, 'o', markersize =4.0)
	plt.plot(xnew, ynew, '-x', alpha=0.25)
	plt.show()


