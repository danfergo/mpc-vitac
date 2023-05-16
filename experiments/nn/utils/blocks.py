from torch import nn
import torch

from functools import reduce


class LinearBNR(nn.Module):
    """
    Combines a linear layer,batch normalization,
    and a ReLU activation function into a single module
    """

    def __init__(self, in_features, out_features):
        super(LinearBNR, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)

        self.batch_norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.linear(x)
        h = self.batch_norm(h)
        h = self.relu(h)
        return h


class DownConv(nn.Module):
    """
    Combines a 2D convolutional layer, 2D batch normalization,
    and a ReLU activation function into a single module
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super(DownConv, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        torch.nn.init.xavier_uniform_(self.conv2d.weight)
        self.conv2d.bias.data.fill_(0.0)

        if activation:
            self.batch_norm = nn.BatchNorm2d(out_channels)

            if activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation.lower() == 'gelu':
                self.activation = nn.GELU()

        # self.batch_norm = nn.BatchNorm2d(out_channels)
        # self.act = nn.ReLU()

    def forward(self, x):
        h = self.conv2d(x)
        if self.activation:
            h = self.batch_norm(h)
            h = self.activation(h)
        return h


class UpConv(nn.Module):
    """
    Combines a 2D transposed convolutional layer, 2D batch normalization,
    and a ReLU activation function into a single module
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='ReLU'):
        super(UpConv, self).__init__()

        self.conv2dTranspose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv2dTranspose.weight)
        self.conv2dTranspose.bias.data.fill_(0.0)

        if activation:
            self.batch_norm = nn.BatchNorm2d(out_channels)

            if activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation.lower() == 'gelu':
                self.activation = nn.GELU()

    def forward(self, x):
        h = self.conv2dTranspose(x)
        if self.activation:
            h = self.batch_norm(h)
            h = self.activation(h)
        return h


### blocks.

class DownBlock(nn.Module):
    """
         Combines a specified number of DownConv layers followed
         by an optional downsampling operation using Max Pooling
    """

    def __init__(self, n_layers, in_channels, hidden_channels=64, out_channels=64, kernel_size=3, down_sample=True,
                 hidden_activation='relu'):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(*[
            DownConv(
                in_channels if i == 0 else hidden_channels,
                out_channels if i == n_layers - 1 else out_channels,
                kernel_size=kernel_size,
                activation=hidden_activation
            )
            for i in range(n_layers)
        ])
        self.down_sample = nn.MaxPool2d(2) if down_sample else None

    def forward(self, x):
        h = self.layers(x)
        if self.down_sample is not None:
            h = self.down_sample(h)
        return h


class UpBlock(nn.Module):
    """
    Combines a specified number of UpConv layers followed
    by an upsampling operation using nearest-neighbor upsampling
    """

    def __init__(self, n_layers, in_channels, hidden_channels=64, out_channels=64, kernel_size=3,
                 up_sample=True, up_sample_before=False, hidden_activation='relu', out_activation='relu'):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(*[
            UpConv(
                in_channels if i == 0 else hidden_channels,
                out_channels if i == n_layers - 1 else hidden_channels,
                kernel_size,
                activation=out_activation if i == n_layers - 1 else hidden_activation
            )
            for i in range(n_layers)
        ])
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2) if up_sample else None
        self.up_sample_before = up_sample_before

    def forward(self, x):
        h = x
        if self.up_sample is not None and self.up_sample_before:
            h = self.up_sample(h)
        h = self.layers(h)
        if self.up_sample is not None and not self.up_sample_before:
            h = self.up_sample(h)
        return h
