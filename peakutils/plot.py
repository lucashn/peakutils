import matplotlib.pyplot as plt

def plot(x, y, indexes):
    plt.plot(x, y, '--', x[indexes], y[indexes], 'r+', ms=5, mew=2)