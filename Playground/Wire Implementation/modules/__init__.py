import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np

pi = np.pi

from scipy.special import expit as sigmoid


def raised_cosine_filter(beta0, T0, num_points=1000):
    lin = np.linspace(-T0, T0, num_points)
    s = 1000
    A = np.ones(lin.shape[0]) * ((1 - beta0) / (2 * T0))
    B = np.ones(lin.shape[0]) * ((1 + beta0) / (2 * T0))
    ABS = np.abs(lin)

    return lin, 1 * sigmoid(s * (A - ABS)) + 0.5 * (
            1 + np.cos(np.pi * T0 * (ABS - ((1 - beta0) / (2 * T0))) / beta0) * (
            sigmoid(s * (B - ABS)) - sigmoid(s * (A - ABS))))


# Parameters
beta = 0.1  # Roll-off factor
T = 1  # Symbol period

# Generate filter valuest\
'''t, rc_filter = raised_cosine_filter(beta, T)

# Plot the filter
plt.plot(t, rc_filter)
plt.title('Raised Cosine Filter')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()'''
