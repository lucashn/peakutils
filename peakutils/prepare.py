'''Data preparation / preprocessing algorithms.'''

def scale(x, new_range=(0., 1.)):
    '''Changes the scale of an array

    Parameters
    ----------
    x : ndarray
        1D array to change the scale (remains unchanged)
    new_range : tuple (float, float)
        Desired range of the array

    Returns
    -------
    ndarray
        Scaled array
    tuple (float, float)
        Previous data range, allowing a rescale to the old range
    '''
    range_ = (x.min(), x.max())
    xp = (x - range_[0]) / (range_[1] - range_[0])
    return xp * (new_range[1] - new_range[0]) + new_range[0], range_
