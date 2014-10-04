import unittest
import peakutils
import os
import numpy
import scipy.signal

def load(name):
    p = os.path.join(os.path.dirname(__file__), name)
    return numpy.loadtxt(p)

class ExperimentalData(unittest.TestCase):
    '''Tests with experimental data'''
    def setUp(self):
        self.data = []
        self.data.append( (load('noise'), 8) )
        self.data.append( (load('baseline'), 2) )

    def test_peaks(self):
        for data in self.data:
            y, n_peaks = data[0][:,1], data[1]
            filtered = scipy.signal.savgol_filter(y, 21, 1)
            idx = peakutils.indexes(filtered, thres=0.08, min_dist=50)

            for p in range(idx.size, 1):
                self.assertGreater(idx[p], 0)
                self.assertLess(idx[p], idx.size-1)
                self.assertGreater(idx[p], idx[p-1])

            self.assertEqual(idx.size, n_peaks)

class SimulatedData(unittest.TestCase):
    '''Tests with simulated data (3 peaks + baseline + nois)'''
    def setUp(self):
        self.x = numpy.linspace(0, 100, 1000)
        self.centers = (20, 40, 70)
        self.y = peakutils.gaussian(self.x, 1, self.centers[0], 3) + \
                 peakutils.gaussian(self.x, 2, self.centers[1], 5) + \
                 peakutils.gaussian(self.x, 3, self.centers[2], 1) + \
                 numpy.random.random(self.x.size) * 0.2

        self.y_base = self.y + numpy.polyval([2., -3., 5.], self.x)

    def test_peaks(self):
        filtered = scipy.signal.savgol_filter(self.y, 51, 3)
        idx = peakutils.indexes(filtered, thres=0.3, min_dist=100)
        peaks = peakutils.interpolate(self.x, self.y, idx, width=30)
        self.assertEqual(idx.size, len(self.centers))
        self.assertEqual(peaks.size, len(self.centers))

        # interpolation should work!
        for i in range(peaks.size):
            self.assertAlmostEqual(peaks[i], self.centers[i], delta=0.5)

    def test_baseline(self):
        y = self.y_base - peakutils.baseline(self.y_base, 2)
        self.assertGreater(numpy.linalg.norm(self.y_base)*0.1,
                           numpy.linalg.norm(y))

        filtered = scipy.signal.savgol_filter(y, 51, 3)
        idx = peakutils.indexes(filtered, thres=0.25, min_dist=100)
        self.assertEqual(idx.size, 3)

        for i, ind in enumerate(idx):
            self.assertAlmostEqual(self.x[ind], self.centers[i], delta=2.)

class Baseline(unittest.TestCase):
    '''Tests the conditioning of the lsqreg in the implementation'''
    def setUp(self):
        self.data = load('exp')

    def test_conditioning(self):
        x, y = self.data[:,0], self.data[:,1]
        mult = 1e-6

        while mult < 100001:
            ny = y * mult
            base = peakutils.baseline(ny, 9) / mult
            self.assertTrue(0.8 < base.max() < 1.0)
            self.assertTrue(-0.1 <= base.min() < 0.1)
            mult *= 10

if __name__ == '__main__':
    numpy.random.seed(1997)
    unittest.main()
