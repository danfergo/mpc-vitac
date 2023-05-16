import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.0):
        super(FourierFeatures, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.b = torch.randn(in_features, out_features) * sigma
        self.pi = 2 * torch.acos(torch.zeros(1)).item()

    def forward(self, x):
        x_proj = 2 * self.pi * x @ self.b
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatures2D(nn.Module):
    def __init__(self, in_channels, out_channels, sigma=1.0):
        super(FourierFeatures2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sigma = sigma

        self.b = (torch.randn(in_channels, out_channels) * sigma).to('cuda')
        self.pi = 2 * torch.acos(torch.zeros(1)).item()

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        print(x.shape)
        x_reshaped = x.reshape(batch_size * height * width * channels, 1)
        b = self.b.view(1, self.in_channels, self.out_channels)
        x_proj_unscaled = x_reshaped @ b
        print(x_proj_unscaled.shape)
        x_proj = 2 * self.pi * x_proj_unscaled
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
