import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np


pi = np.pi

from scipy.special import expit as sigmoid



def raised_cosine_filter(beta0, T0, num_points=1000):
    lin = torch.linspace(-T0, T0, num_points)
    m = nn.Threshold(0, 0)
    # s = 1000
    s0 = 80
    f1 = (1 - beta0) / (2 * T0)
    f2 = (1 + beta0) / (2 * T0)
    f_ = T0 / 2 * (1 + torch.cos(torch.pi * T0 / beta0 * (lin - f1)))

    out = T0 * (torch.sigmoid((lin)) - torch.sigmoid(s0 * (lin - f1))) \
          + f_ * (torch.sigmoid(s0 * (f1 - lin)) - torch.sigmoid(s0 * (f2 - lin)))

    return lin, out

    # A = torch.ones(lin.shape[0]) * ((1 - beta0) / (2 * T0))
    # B = torch.ones(lin.shape[0]) * ((1 + beta0) / (2 * T0))
    # ABS = torch.abs(lin)
    #
    # # return (1+)
    #
    # return (lin, 1 * (m(ABS)-m(ABS-A))
    #         + T0*0.5*(1 + torch.cos(np.pi * T0/beta0 * (ABS - ((1 - beta0) / (2 * T0)))) * (m(ABS-A) - m(ABS-B))))


# Parameters
beta = 0.3  # Roll-off factor
T = 1  # Symbol period

# Generate filter valuest\
t, rc_filter = raised_cosine_filter(beta, T)

# Plot the filter
plt.plot(t, rc_filter)
plt.title('Raised Cosine Filter')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


print(np.ones(10) + 1)


