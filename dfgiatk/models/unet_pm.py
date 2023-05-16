import torch

from torch import nn
from functools import partial


def unet_pm(out_channels=1, a_channels=6, weights=None, pretrained=True):
    """
        Dynamics model ot,at -> ot+1 based on unet.
        https://github.com/milesial/Pytorch-UNet#pretrained-model
    """
    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=pretrained, scale=0.5)

    # model.a1_conv = nn.Conv2d(a_channels, 64, kernel_size=1)
    # model.a2_conv = nn.Conv2d(a_channels, 128, kernel_size=1)
    # model.a3_conv = nn.Conv2d(a_channels, 256, kernel_size=1)
    # model.a4_conv = nn.Conv2d(a_channels, 512, kernel_size=1)
    # model.a5_conv = nn.Conv2d(a_channels, 1024, kernel_size=1)

    model.a1add_enc = nn.Linear(196 + 6, 196)
    model.a1add_dec = nn.Conv2d(1, 1024, kernel_size=1)

    model.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    # print(model)

    def forward(self, x, a):
        # a_transposed = a[:, :, None, None]

        # a sizes: e.g. [8, 6]
        # x's sizes: x1.size(), x2.size(), x3.size(), x4.size(), x5.size()
        # [8, 64, 224, 224], [8, 128, 112, 112], [8, 256, 56, 56], [8, 512, 28, 28] [8, 1024, 14, 14])

        # encode action into 2D tensors
        # a1 = self.a1_conv(a_transposed.repeat(1, 1, 224, 224))
        # a2 = self.a2_conv(a_transposed.repeat(1, 1, 112, 112))
        # a3 = self.a3_conv(a_transposed.repeat(1, 1, 56, 56))
        # a4 = self.a4_conv(a_transposed.repeat(1, 1, 28, 28))
        # a5 = self.a5_conv(a_transposed.repeat(1, 1, 14, 14))

        # downstream from original u-net
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # action merge
        # x1 = x1 + a1
        # x2 = x2 + a2
        # x3 = x3 + a3
        # x4 = x4 + a4
        # x5 = x5 + a5

        x5, _ = torch.max(x5, dim=1)
        x5_flat = torch.reshape(x5, (-1, 14 * 14))
        x5_flat = torch.cat((x5_flat, a), 1)
        x5 = self.a1add_enc(x5_flat)
        x5 = torch.reshape(x5, (-1, 1, 14, 14))
        x5 = self.a1add_dec(x5)

        # upstream from original u-net
        x = self.up1(x5, x4 * 0)
        x = self.up2(x, x3 * 0)
        x = self.up3(x, x2 * 0)
        x = self.up4(x, x1 * 0)

        logits = self.outc(x)

        return logits

    setattr(model, 'forward', partial(forward, model))

    if weights is not None:
        print('laded weights', weights)
        model.load_state_dict(torch.load(weights))

    return model
