import os
import unittest
import warnings
import itertools

import numpy
from numpy.testing import assert_array_almost_equal
import scipy.signal
import numpy as np
import pandas as pd
import peakutils

def load(name):
    p = os.path.join(os.path.dirname(__file__), name)
    return numpy.loadtxt(p)


class LPGPeaks(unittest.TestCase):

    """Tests with experimental data from long period gratings"""

    def test_peaks(self):
        y = load('noise')[:, 1]
        filtered = scipy.signal.savgol_filter(y, 21, 1)
        n_peaks = 8

        idx = peakutils.indexes(filtered, thres=0.08, min_dist=50)

        for p in range(idx.size, 1):
            self.assertGreater(idx[p], 0)
            self.assertLess(idx[p], idx.size - 1)
            self.assertGreater(idx[p], idx[p - 1])

        self.assertEqual(idx.size, n_peaks)


class FBGPeaks(unittest.TestCase):

    """Tests with experimental data from fiber Bragg gratings"""

    def test_peaks(self):
        data = load('baseline')
        x, y = data[:, 0], data[:, 1]
        n_peaks = 2

        prepared = y - peakutils.baseline(y, 3)
        idx = peakutils.indexes(prepared, thres=0.03, min_dist=5)

        for p in range(idx.size, 1):
            self.assertGreater(idx[p], 0)
            self.assertLess(idx[p], idx.size - 1)
            self.assertGreater(idx[p], idx[p - 1])

        self.assertEqual(idx.size, n_peaks)
        assert_array_almost_equal(x[idx], numpy.array([1527.3, 1529.77]))


class SimulatedData(unittest.TestCase):

    """Tests with simulated data"""

    def setUp(self):
        self.near = numpy.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])

    def aux_test_peaks(self, dtype):
        """(3 peaks + baseline + noise)"""
        x = numpy.linspace(0, 100, 1000)
        centers = (20, 40, 70)
        y = (1000 * (peakutils.gaussian(x, 1, centers[0], 3) +
             peakutils.gaussian(x, 2, centers[1], 5) +
             peakutils.gaussian(x, 3, centers[2], 1) +
             numpy.random.random(x.size) * 0.2)).astype(dtype)

        filtered = scipy.signal.savgol_filter(y, 51, 3).astype(dtype)
        
        idx = peakutils.indexes(filtered, thres=0.3, min_dist=100)
        peaks = peakutils.interpolate(x, y, idx, width=30)
        self.assertEqual(idx.size, len(centers))
        self.assertEqual(peaks.size, len(centers))

        # interpolation should work!
        for i in range(peaks.size):
            self.assertAlmostEqual(peaks[i], centers[i], delta=0.5)

    def test_peaks(self):
        self.aux_test_peaks('float64')
        self.aux_test_peaks('float32')
        self.aux_test_peaks('int32')
        self.assertRaises(ValueError, self.aux_test_peaks, 'uint32')

    def test_near_peaks1(self):
        out = peakutils.indexes(self.near, thres=0, min_dist=2)
        expected = numpy.array([1, 5, 9])
        assert_array_almost_equal(out, expected)

    def test_near_peaks2(self):
        out = peakutils.indexes(self.near, thres=0, min_dist=1)
        expected = numpy.array([1, 3, 5, 7, 9])
        assert_array_almost_equal(out, expected)

    def test_list_peaks(self):
        out = peakutils.indexes([1, 2, 1, 3, 5, 7, 4, 1], thres=0, min_dist=1)
        expected = numpy.array([1, 5])
        assert_array_almost_equal(out, expected)

    def test_pandas_series(self):
        x = ["a", "b", "c", "d", "e"]
        y = [  0,   2,   0,  3,   0 ]
        data = pd.Series(data=y, index=x)
        out = peakutils.indexes(data, thres=0, min_dist=1)
        expected = numpy.array([1, 3])
        assert_array_almost_equal(out, expected)
    
    def test_absolute_threshold(self):
        x = [0, 5, 0, 8, 0, 15, 0]
        out1 = peakutils.indexes(x, thres=3, thres_abs=True)
        assert_array_almost_equal(out1, [1, 3, 5])

        out2 = peakutils.indexes(x, thres=5, thres_abs=True)
        assert_array_almost_equal(out2, [3, 5])

        out3 = peakutils.indexes(x, thres=7, thres_abs=True)
        assert_array_almost_equal(out3, [3, 5])

        out4 = peakutils.indexes(x, thres=14, thres_abs=True)
        assert_array_almost_equal(out4, [5])

        out5 = peakutils.indexes(x, thres=15, thres_abs=True)
        assert_array_almost_equal(out5, [])

        out6 = peakutils.indexes(x, thres=16, thres_abs=True)
        assert_array_almost_equal(out6, [])

class Baseline(unittest.TestCase):

    """Tests the conditioning of the lsqreg in the implementation"""

    def test_conditioning(self):
        data = data = load('exp')
        y = data[:, 1]
        mult = 1e-6

        while mult < 100001:
            ny = y * mult
            base = peakutils.baseline(ny, 9) / mult
            self.assertTrue(0.8 < base.max() < 1.0)
            self.assertTrue(-0.1 <= base.min() < 0.1)
            mult *= 10

    def test_negative(self):
        data = np.array([-1, -2, -3, -4, -3, -2, -1] * 10)
        base = peakutils.baseline(data)
        self.assertEqual(data.shape, base.shape)

class Prepare(unittest.TestCase):

    """Tests the prepare module"""

    def test_scale(self):
        orig = numpy.array([-2, -1, 0.5, 1, 3])
        x1, range_old = peakutils.scale(orig, (-10, 8))
        x2, range_new = peakutils.scale(x1, range_old)

        assert_array_almost_equal(orig, x2)
        self.assertTupleEqual(range_new, (-10, 8))

    def test_scale_degenerate(self):
        orig = numpy.array([-3, -3, -3])
        x1, range_old = peakutils.scale(orig, (5, 7))
        x2, range_new = peakutils.scale(x1, range_old)

        assert_array_almost_equal(x1, [6, 6, 6])
        assert_array_almost_equal(x2, orig)

class Centroid(unittest.TestCase):

    """Tests the centroid implementations."""

    def test_centroid(self):
        y = np.ones(10)
        x = np.arange(10)
        self.assertEqual(peakutils.centroid(x, y), 4.5)

    def test_centroid2(self):
        y = np.ones(3)
        x = np.array([0., 1., 9.])
        c, v = peakutils.centroid2(y, x)
        self.assertEqual(c, 4.5)

class GaussianFit(unittest.TestCase):

    """ Tests the Gaussian fit implementation """

    def test_gaussian_fit(self):
        params = np.array([0.5, 6, 2])
        x = np.arange(10)
        y = peakutils.gaussian(x, *params)
        self.assertAlmostEqual(peakutils.gaussian_fit(x, y), params[1])

        res = peakutils.gaussian_fit(x, y, center_only=False)
        np.testing.assert_allclose(res, params)

class Plateau(unittest.TestCase):

    """Issue #4"""

    def test_plateau1(self):
        y = np.zeros(20)
        y[1:6] = 1.0
        y[8:9] = 2.0
        y[11:19] = 3.0
        idx = peakutils.indexes(y)
        np.testing.assert_array_equal(idx, [3, 8, 14])

    def test_plateau2(self):
        y = np.zeros(20)
        y[0:6] = 1.0
        y[8:9] = 2.0
        y[11:20] = 3.0
        idx = peakutils.indexes(y)
        np.testing.assert_array_equal(idx, [8])
        # note: there are no peaks in the first and last series as the data
        # to the left of 0 and right of 19 is unknown
        
    def test_flat(self):
        ra = (0.2, 0.4, 0.6, 0.8, 0.95)
        rb = (1, 2, 3, 4, 5, 6)
        N = 20
        
        # all equal
        for t, m in itertools.product(ra, rb):
            y = np.ones(N)
            peakutils.indexes(y, thres=t, min_dist=m)
            
        # a single peak
        for t, m in itertools.product(ra, rb):
            for z in range(m + 1, N - m):
                y = np.ones(N)
                y[z] = 1e3
                p = peakutils.indexes(y, thres=t, min_dist=m)
                self.assertEqual(p, np.array([z]))

class Float64(unittest.TestCase):

    """Issue #11 (false alarm)"""
    def setUp(self):
        self.col = [
            u'2161', u'183', u'167', u'270', u'164', u'475', u'327', u'279', u'0', 
            u'183', u'360', u'81', u'81', u'81', u'81', u'45', u'81', u'0', u'81', u'81'
        ]

    def test_int_high_thres(self):
        y = np.atleast_1d(self.col).astype('int')
        peaks = peakutils.indexes(y, thres=0.3)
        np.testing.assert_array_almost_equal(peaks, [])

    def test_float64_high_thres(self):
        y = np.atleast_1d(self.col).astype('float64')
        peaks = peakutils.indexes(y, thres=0.3)
        np.testing.assert_array_almost_equal(peaks, [])
        
    def test_int_low_thres(self):
        y = np.atleast_1d(self.col).astype('int')
        peaks = peakutils.indexes(y, thres=0.01)
        np.testing.assert_array_almost_equal(peaks, [3, 5, 10, 16])

    def test_float64_low_thres(self):
        y = np.atleast_1d(self.col).astype('float64')
        peaks = peakutils.indexes(y, thres=0.01)
        np.testing.assert_array_almost_equal(peaks, [3, 5, 10, 16])

class InterpolateExceptions(unittest.TestCase):
    
    """ Issue #14: convert fitting errors to warnings """
    def test_interpolate_bounds(self):
        x = np.arange(5)
        y = np.array([0, 0, 1, 0, 0])
        
        with warnings.catch_warnings(record=True) as record:
            for w in range(1, 10):
                peakutils.interpolate(x, y, [2], width=w)
        
        self.assertGreater(len(record), 0)
        
class HighEnvelope(unittest.TestCase):
    
    def test_up_envelope(self):
        data = np.array([0, 2, 0, 0, 4, 0, 0, 0, 7, 0, 0, 0, 0, 9, 0, 0, 0, 11, 0, 0, 0, 0, 0, 9, 0,
                         0, 0, 0, 7, 0, 0, 4, 0, 0, 3, 0, 0, 0, 1])
        env = peakutils.envelope(data, 5)
        tol = 1.05
        
        for a, b in zip(data, env):
            self.assertLess(a, b * tol)

if __name__ == '__main__':
    numpy.random.seed(1997)
    unittest.main()
