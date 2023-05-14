# Author: Gyan Tatiya

import numpy as np

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, h_dims, out_dim, h_activ=None, out_activ=None):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        num_layers = len(layer_dims) - 1
        layers = []
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index], layer_dims[index + 1])

            if h_activ and index < num_layers - 1:
                layers.extend([layer, h_activ])  # nn.Dropout()
            elif out_activ and index == num_layers - 1:
                layers.extend([layer, out_activ])
            else:
                layers.append(layer)

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, h_dims, out_dim, h_activ=None, out_activ=None):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        num_layers = len(layer_dims) - 1
        layers = []
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index], layer_dims[index + 1])

            if h_activ and index < num_layers - 1:
                layers.extend([layer, h_activ])  # nn.Dropout()
            elif out_activ and index == num_layers - 1:
                layers.extend([layer, out_activ])
            else:
                layers.append(layer)

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class EncoderDecoderNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_layer_sizes=[], n_dims_code=2, h_activation_fn=None,
                 out_activation_fn=None):
        super(EncoderDecoderNetwork, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.h_activation_fn = h_activation_fn
        self.out_activation_fn = out_activation_fn

        self.encoder = Encoder(self.input_channels, self.hidden_layer_sizes, self.n_dims_code,
                               h_activ=self.h_activation_fn, out_activ=self.h_activation_fn)
        self.decoder = Decoder(self.n_dims_code, list(reversed(self.hidden_layer_sizes)), self.output_channels,
                               h_activ=self.h_activation_fn, out_activ=self.out_activation_fn)

    def forward(self, x):

        z = self.encoder(x)
        x = self.decoder(z)

        return x, z
