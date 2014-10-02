import numpy as np
import peakutils
from timeit import default_timer as timer

np.random.seed(1997)

def make_data(n):
    x = np.linspace(0, 100, n)
    y = peakutils.gaussian(x, 5, 20, 10) + \
        peakutils.gaussian(x, 8, 70, 3)  + \
        np.random.rand(x.size) + np.polyval([3,2], x)
    return x, y

def benchit():
    tests = [("Small", make_data(200), 100), ("Big", make_data(20000), 5)]

    for name, data, rep in tests:
        begin = timer()

        for _ in range(rep):
            y = data[1] - peakutils.baseline(data[1])
            i = peakutils.indexes(y, thres=0.3, min_dist=y.size / 10)
            _ = peakutils.interpolate(data[0], y, i)

        end = timer()
        each = (end-begin) / rep
        print("*{}* test took {:.3f} seconds each rep".format(name, each))

if __name__ == '__main__':
    benchit()
