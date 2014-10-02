import numpy as np
from numpy.linalg import norm

def baseline(y, deg=3, max_it=100, tol=1e-3):
    '''Computes the baseline of a given data.

    Iteratively performs a polynomial fitting in the data to detect its
    baseline. At every iteration, the fitting weights on the regions with
    peaks is reduced to identify the baseline only.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    deg : int
        Degree of the polynomial that will estimate the data baseline. A low
        degree may fail to detect all the baseline present, while a high
        degree may make the data too oscillatory, especially at the edges.
    max_it : int
        Maximum number of iterations to perform.
    tol : float
        Tolerance to use when comparing the difference between the current
        fit coefficient and the ones from the last iteration. The iteration
        procedure will stop when the difference between them is lower than
        *tol*.

    Returns
    -------
    ndarray
        Array with the amplitude of the baseline for every original point in *y*
    '''
    coeffs = np.ones(deg+1)
    x = np.arange(y.size)
    base = y.copy()

    for it in range(max_it):
        coeffs_new = np.polyfit(x, y, deg)

        if norm(coeffs_new-coeffs) / norm(coeffs) < tol:
            break

        coeffs = coeffs_new
        base = np.polyval(coeffs, x)
        y = np.minimum(y, base)

    return base