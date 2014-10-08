import unittest
import peakutils
import os
import numpy
from numpy.testing import assert_array_almost_equal
import scipy.signal

def load(name):
    p = os.path.join(os.path.dirname(__file__), name)
    return numpy.loadtxt(p)

class ExperimentalData(unittest.TestCase):
    '''Tests with experimental data'''
    def test_peaks(self):
        cases = [(load('noise'), 8), (load('baseline'), 2)]
        
        for data in cases:
            y, n_peaks = data[0][:,1], data[1]
            filtered = scipy.signal.savgol_filter(y, 21, 1)
            idx = peakutils.indexes(filtered, thres=0.08, min_dist=50)

            for p in range(idx.size, 1):
                self.assertGreater(idx[p], 0)
                self.assertLess(idx[p], idx.size-1)
                self.assertGreater(idx[p], idx[p-1])

            self.assertEqual(idx.size, n_peaks)

class SimulatedData(unittest.TestCase):
    '''Tests with simulated data'''
    
    def test_peaks(self):
        '''(3 peaks + baseline + noise)'''
        x = numpy.linspace(0, 100, 1000)
        centers = (20, 40, 70)
        y = (peakutils.gaussian(x, 1, centers[0], 3) + 
             peakutils.gaussian(x, 2, centers[1], 5) +
             peakutils.gaussian(x, 3, centers[2], 1) +
             numpy.random.random(x.size) * 0.2)

        y_base = y + numpy.polyval([2., -3., 5.], x)
        
        filtered = scipy.signal.savgol_filter(y, 51, 3)
        idx = peakutils.indexes(filtered, thres=0.3, min_dist=100)
        peaks = peakutils.interpolate(x, y, idx, width=30)
        self.assertEqual(idx.size, len(centers))
        self.assertEqual(peaks.size, len(centers))

        # interpolation should work!
        for i in range(peaks.size):
            self.assertAlmostEqual(peaks[i], centers[i], delta=0.5)

    def test_near_peaks(self):
        y = numpy.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
        out = peakutils.indexes(y, thres=0, min_dist=2)
        expected = numpy.array([3, 5, 9])
        assert_array_almost_equal(out, expected)

class Baseline(unittest.TestCase):
    '''Tests the conditioning of the lsqreg in the implementation'''
    def test_conditioning(self):
        data = data = load('exp')
        _, y = data[:,0], data[:,1]
        mult = 1e-6

        while mult < 100001:
            ny = y * mult
            base = peakutils.baseline(ny, 9) / mult
            self.assertTrue(0.8 < base.max() < 1.0)
            self.assertTrue(-0.1 <= base.min() < 0.1)
            mult *= 10

class Prepare(unittest.TestCase):
    '''Tests the prepare module'''
    def test_scale(self):
        orig = numpy.array([-2, -1, 0.5, 1, 3])
        x1, range_old = peakutils.scale(orig, (-10, 8))
        x2, range_new = peakutils.scale(x1, range_old)

        assert_array_almost_equal(orig, x2)
        self.assertTupleEqual(range_new, (-10, 8))
        
    def test_scale_degenerate(self):
        orig = numpy.array([-3,-3,-3])
        x1, range_old = peakutils.scale(orig, (5, 7))
        x2, range_new = peakutils.scale(x1, range_old)
        
        assert_array_almost_equal(x1, [6,6,6])
        assert_array_almost_equal(x2, orig)

if __name__ == '__main__':
    numpy.random.seed(1997)
    unittest.main()
