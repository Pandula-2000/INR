
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

pi = np.pi


class RaisedCosineLayer(nn.Module):
    '''
        Implicit representation with Raised Cosine activation.

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 beta0=0.5,
                 T0=0.1,
                 trainable=False):
        super().__init__()
        self.beta0 = beta0
        self.T0 = T0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features

        if self.is_first:
            dtype = torch.float  # 32 bit
        else:
            dtype = torch.cfloat  # complex 64 bit

        # Set trainable parameters if they are to be simultaneously optimized
        self.T0 = nn.Parameter(self.T0 * torch.ones(1), trainable)
        self.beta0 = nn.Parameter(self.beta0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

        # # Second Gaussian window
        # self.scale_orth = nn.Linear(in_features,
        #                             out_features,
        #                             bias=bias,
        #                             dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        # beta = self.beta0 * lin
        # T = self.T0 * lin
        A = torch.ones(input.shape[0])*(1-self.beta0)/2*self.T0
        B = torch.ones(input.shape[0])*(1+self.beta)/2*self.T0
        ABS = lin.abs()

        return 1 * torch.heaviside(A - ABS, torch.tensor([0])) + 0.5 * (1 + torch.cos(pi * self.T0 * (ABS - (1 - self.beta0 / 2 * self.T0)) / self.beta0) * (torch.heaviside(B - ABS, torch.tensor([0])) - torch.heaviside(A - ABS, torch.tensor([0]))))


class INR(nn.Module):
    def __init__(self, in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_beta=0.3,
                 hidden_beta=0.5,
                 T=0.1,
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = RaisedCosineLayer

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 2 (NOTE: Skipped this)
        # hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        # self.complex = True
        # self.wavelet = 'gabor'

        # Legacy parameter
        self.pos_encode = False

        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features,
                                    beta0=first_beta,
                                    T0=T,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        beta0=hidden_beta,
                                        T0=T))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        # if self.wavelet == 'gabor':
        #     return output.real
        return output











