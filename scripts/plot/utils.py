import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def exponential_moving_average(func, x_min, x_max, n, scale):
    x = np.linspace(x_min, x_max, n)
    y = np.zeros(n)
    decay_factor = np.exp(-(x[1] - x[0])/scale)
    for i in range(n):
        y[i] = decay_factor*y[i - 1] + (1. - decay_factor)*func(x[i]) if i > 0 else func(x[0])
    return x, y


def plot_quantity_in_axis(x, y, ax, label = None, smoothing_scale = None, 
        n = 1000, color = None, linestyle = "-"):
    """Plot quantity in given axis."""

    # Note: a return is only given at the end of every episode. Hence, if the
    # x-axis does not show the number of episodes but, e.g., timesteps, interpolation
    # is used in order to be able to calculate a mean curve
    return_funcs = []

    return_func = interp1d(x, y)
    x_min = np.min(x)
    x_max = np.max(x)

    if smoothing_scale is not None:
        x_run, y_run = exponential_moving_average(return_func, x_min, x_max, n, 
                smoothing_scale)
    else:
        x_run = x
        y_run = y
    ax.plot(x_run, y_run, color=color, linestyle=linestyle, label=label)


def plot_success(data, smoothing_scale):
    """Plot success rate during training."""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("steps")
    ax.set_ylabel("success rate")
    steps = data
