import numpy as np
import scipy.linalg as LA

def baseline(y, deg=3, max_it=10, tol=1e-3):
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
        Array with the amplitude of the baseline for every point in *y*
    '''
    coeffs = np.ones(deg+1) # initial coefficient estimate
    base = y.copy()         # current baseline estimate

    # speed up the computation by computing the vandermonde and its pinv once
    x = np.arange(y.size)
    vander = np.vander(x, deg+1)
    vander_pinv = LA.pinv2(vander)

    for it in range(max_it):
        coeffs_new = np.dot(vander_pinv, y)
        base = np.dot(vander, coeffs_new)

        if LA.norm(coeffs_new-coeffs) / LA.norm(coeffs) < tol:
            break

        coeffs = coeffs_new
        y = np.minimum(y, base)

    return base