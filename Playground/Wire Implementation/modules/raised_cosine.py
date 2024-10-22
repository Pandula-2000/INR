import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

pi = np.pi
sigmoid = torch.nn.Sigmoid()


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
                 s0=1000,
                 bias=True,
                 is_first=False,
                 beta0=0.5,
                 T0=0.6,
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
        self.T0 = nn.Parameter(self.T0 * torch.ones(1), trainable)
        self.beta0 = nn.Parameter(self.beta0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):

        # lin = self.linear(input).abs().cuda()
        lin = input.cuda()

        abs_lin = torch.abs(lin)  # Renamed to avoid shadowing 'abs' function

        f1 = (1 - self.beta0) / (2 * self.T0)
        f2 = (1 + self.beta0) / (2 * self.T0)

        # f_ = self.T0 / 2 * (1 + torch.cos(torch.pi * self.T0 / self.beta0 * (abs_lin - f1)))
        f_ = 1 / 2 * (1 + torch.cos(torch.pi * self.T0 / self.beta0 * (abs_lin - f1)))
        # out = self.T0 * (sigmoid(self.s0 * (abs_lin)) - sigmoid(self.s0 * (abs_lin - f1))) \
        #       + f_ * (sigmoid(self.s0 * (abs_lin - f1)) - sigmoid(self.s0 * (abs_lin - f2)))

        out = 1 * (sigmoid(self.s0 * (abs_lin)) - sigmoid(self.s0 * (abs_lin - f1))) \
              + f_ * (sigmoid(self.s0 * (abs_lin - f1)) - sigmoid(self.s0 * (abs_lin - f2)))

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
                 scale=None,
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


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    lin = torch.linspace(-2, 2, 100).cuda()
    o = RaisedCosineLayer(100, 100)
    o.cuda()
    # Forward pass through the layer
    output = o(lin)

    # Convert the output to a NumPy array
    output_np = output.detach().cpu().numpy()

    # Plot the output
    plt.plot(lin.detach().cpu().numpy(), output_np)
    plt.title('Raised Cosine Filter')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
