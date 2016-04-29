import numpy as np
import peakutils
from timeit import default_timer as timer

def make_data(n, noise):
    x = np.linspace(0, 100, n)
    y = peakutils.gaussian(x, 5, 20, 10) + \
        peakutils.gaussian(x, 8, 70, 3) + \
        noise * np.random.rand(x.size) + np.polyval([3, 2], x)
    return x, y


def benchit():
    tests = [("Small - Low noise",  make_data(200, 1.), 100),
             ("Small - High noise", make_data(200, 3.), 100),
             ("Big - Low noise",    make_data(20000, 1), 5),
             ("Big - High noise",   make_data(20000, 2.), 5)]

    for name, data, rep in tests:
        begin = timer()

        for _ in range(rep):
            y = data[1] - peakutils.baseline(data[1])
            i = peakutils.indexes(y, thres=0.4, min_dist=y.size // 5)

        end = timer()
        each = (end - begin) / rep
        print("*{}* test took {:.3f} seconds each rep".format(name, each))

if __name__ == '__main__':
    np.random.seed(1997)
    benchit()
