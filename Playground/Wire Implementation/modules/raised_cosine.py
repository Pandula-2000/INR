
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
                 s0 = 1,
                 bias=True,
                 is_first=False,
                 beta0=0.5,
                 T0=0.1,
                 trainable=False):
        super().__init__()
        self.s0 = s0
        self.beta0 = beta0
        self.T0 = T0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features

        dtype = torch.float  # complex 64 bit

        # Set trainable parameters if they are to be simultaneously optimized
        self.s0 = nn.Parameter(self.s0 * torch.ones(1), trainable)
        self.T0 = nn.Parameter(self.T0 * torch.ones(1), trainable)
        self.beta0 = nn.Parameter(self.beta0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        abs_lin = lin.abs().cuda()
        
        f1 = (1 - self.beta0) / (2 * self.T0)
        f2 = (1 + self.beta0) / (2 * self.T0)        
        f_ = self.T0 / 2 * (1 + torch.cos(torch.pi * self.T0 / self.beta0 * (abs_lin - f1)))

        out = self.T0 * (torch.sigmoid(self.s0*(abs_lin)) - torch.sigmoid(self.s0*(abs_lin - f1))) \
        + f_ * (torch.sigmoid(self.s0*(abs_lin - f1)) - torch.sigmoid(self.s0*(abs_lin - f2)))
        
        return out
        
class INR(nn.Module):
    def __init__(self, in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 beta0=0.5,
                 T0=0.1,
                 sclae=None,
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        self.nonlin = RaisedCosineLayer
        dtype = torch.float

        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features,
                                    beta0=beta0,
                                    T0=T0,
                                    is_first=True,
                                    trainable=True))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        beta0=beta0,
                                        T0=T0,
                                        is_first=True,
                                        trainable=True))

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)        
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output











