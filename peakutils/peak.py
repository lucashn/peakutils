import numpy as np
from scipy import optimize

def indexes(y, thres=0.3, min_dist=1):
    '''Peak detection routine.

    Finds the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks.

    Parameters
    ----------
    y : ndarray
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the indexes of the peaks that were detected
    '''
    thres *= np.max(y) - np.min(y)

    # find the peaks by using the first order difference
    dy = np.diff(y)
    peaks = np.where((np.hstack([dy,0.])<0.)
                    & (np.hstack([0.,dy])>0.)
                    & (y > thres) )[0]

    if peaks.size > 1 and min_dist > 1:
        new_peaks = []
        top = peaks[0]
        padding = y.size+min_dist+1

        for p in np.append(peaks,padding):
            if (p - top) <= min_dist: # still in the same group?
                if y[p] > y[top]:
                    top = p
            else: # end, start another 'group'
                new_peaks.append(top)
                top = p

        peaks = np.array(new_peaks)

    return peaks

def centroid(x, y):
    '''Computes the centroid for the specified data.

    Parameters
    ----------
    x : ndarray
        Data on the x axis.
    y : ndarray
        Data on the y axis.

    Returns
    -------
    float
        Centroid of the data.
    '''
    return np.sum(x*y)/np.sum(y)

def gaussian(x, a, b, c):
    '''Computes the Gaussian function.

    Parameters
    ----------
    x : float
        Point to evaluate the Gaussian for.
    a : float
        Amplitude.
    b : float
        Center.
    c : float
        Width.

    Returns
    -------
    float
        Value of the specified Gaussian at *x*
    '''
    return a * np.exp(-(x-b)**2 / (2*c**2))

def gaussian_fit(x, y):
    '''Performs a Gaussian fitting of the specified data.

    Parameters
    ----------
    x : ndarray
        Data on the x axis.
    y : ndarray
        Data on the y axis.

    Returns
    -------
    ndarray
        Parameters of the Gaussian that fits the specified data
    '''
    r, o = optimize.curve_fit(gaussian, x, y, [np.max(y), x[0], (x[1]-x[0])*5])
    return r[1]

def interpolate(x, y, ind=None, width=10, func=gaussian_fit):
    '''Tries to enhance the resolution of the peak detection by using
    Gaussian fitting, centroid computation or an arbitrary function on the
    neighborhood of each previously detected peak index.

    Parameters
    ----------
    x : ndarray
        Data on the x dimension.
    y : ndarray
        Data on the y dimension.
    ind : ndarray
        Indexes of the previously detected peaks. If None, indexes() will be
        called with the default parameters.
    width : int
        Number of points (before and after) each peak index to pass to *func*
        in order to encrease the resolution in *x*.
    func : function(x,y)

    Returns
    -------
    ndarray :
        Array with the adjusted peak positions (in *x*)
    '''

    if ind is None:
        ind = indexes(y)

    out = []
    for sl in (slice(i-width, i+width) for i in ind):
        try:
            fit = func(x[sl], y[sl])
            out.append(fit)
        except:
            pass

    return np.array(out)
