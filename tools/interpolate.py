import numpy as np
import matplotlib.pyplot as plt


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




input_filename = '../gps2/15_mod_car2.csv'
output_filename = '../gps_interpolated/15_intp_car2.csv'



times, speeds = data_from_csv(input_filename)
if len(speeds) != len(times):
    print "Speeds array not same length as times array!"
    exit()

xnew = np.arange(times[0], times[-1], 33.3666666667)
ynew = np.interp(xnew, times, speeds)

# plt.plot(times, speeds, 'o', markersize =4.0)
# plt.plot(xnew, ynew, '-x', alpha=0.25)
# plt.show()
if len(xnew) != len(ynew):
    print "Array lenght mismatch, theres a bug in the code!"
    exit()

out = open(output_filename, 'w')
for i in range(len(xnew)):
    out.write( str(xnew[i]) + "," + str(ynew[i]) + '\n')
out.close()



