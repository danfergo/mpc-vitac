import torch
from torch import nn


def unet(n_channels=1, weights=None):
    """
        Image Encoder/Decoder, U-Net
        https://github.com/milesial/Pytorch-UNet#pretrained-model
    """
    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    model.outc = nn.Conv2d(64, n_channels, kernel_size=1)

    if weights is not None:
        model.load_state_dict(torch.load(weights))

    return model
