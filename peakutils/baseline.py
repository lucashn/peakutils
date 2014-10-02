import numpy as np
from numpy.linalg import norm

def baseline(y, deg=3, max_it=100, tol=1e-3):
    coeffs = np.zeros(deg+1)
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