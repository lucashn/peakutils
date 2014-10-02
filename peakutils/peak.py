import numpy as np
from scipy import optimize

def peak_indexes(y, min_delta=0.01, thres=0.05, min_dist=1):
    # normalized min_delta to actual min_delta
    range_ = np.max(y) - np.min(y)
    min_delta *= range_

    # find the peaks by using the first order difference
    dy = np.diff(y)
    peaks = np.where((np.hstack([dy,0.])<-min_delta)
                    & (np.hstack([0.,dy])>min_delta)
                    & (y > thres) )[0]

    if peaks.size > 1 and min_dist > 1:
        new_peaks = []
        group = [peaks[0]]
        last = peaks[0]
        padding = y.size+min_dist+1

        for p in np.append(peaks,padding):
            if (p - last) <= min_dist:
                # still in the same group
                group.append(p)
            else:
                # end, start another group
                top = group[np.argsort(y[group])[-1]] # highest element
                new_peaks.append(top)
                group = [p]

            last = p
        peaks = np.array(new_peaks)

    return peaks

def centroid(x, y):
    return np.sum(x*y)/np.sum(y)

def gaussian(x, a, b, c):
    return a * np.exp(-(x-b)**2 / c)

def gaussian_fit(x, y):
    r, o = optimize.curve_fit(gaussian, x, y, [np.max(y), x[0], (x[1]-x[0])*5])
    return r[1]

def peak_estimate(x, y, indexes=None, width=10, func=gaussian_fit):
    if indexes is None:
        indexes = peak_indexes(y)

    return np.array([func(x[s], y[s]) for s in
            (slice(i-width, i+width) for i in indexes)])