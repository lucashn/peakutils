import matplotlib.pyplot as plt


def plot(x, y, ind):
    """
    Plots the original data with the peaks that were identified

    Parameters
    ----------
    x : array-like
        Data on the x-axis
    y : array-like
        Data on the y-axis
    ind : array-like
        Indexes of the identified peaks
    """
    plt.plot(x, y, "--")

    marker_x = x.iloc[ind] if hasattr(x, "iloc") else x[ind]
    marker_y = y.iloc[ind] if hasattr(y, "iloc") else y[ind]

    plt.plot(marker_x, marker_y, "r+", ms=5, mew=2, label="{} peaks".format(len(ind)))
    plt.legend()
