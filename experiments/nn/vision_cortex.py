from collections import namedtuple

from torch import nn

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import load_weights, run_sequentially

from experiments.nn.utils.fourier_features import FourierFeatures, FourierFeatures2D

VisionCortexSize = namedtuple('VisualCortexSize', ['n_blocks', 'block_size', 'channels'])


class VisionCortex(nn.Module):

    def __init__(self, size: VisionCortexSize = VisionCortexSize(3, 3, 64), weights=None, use_fourier_features=True):
        super(VisionCortex, self).__init__()

        channels = size.channels

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=size.block_size,
                in_channels=3 if i == 0 else channels,
                hidden_channels=channels,
                out_channels=channels
            )
            for i in range(size.n_blocks)
        ])

        # self.fourier_features = None
        # if use_fourier_features:
        #     self.fourier_features = FourierFeatures2D(channels, channels)
        #     channels = 2 * channels

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=size.block_size,
                in_channels=channels,
                hidden_channels=channels,
                out_channels=3 if (i == size.n_blocks) else channels,
                # up_sample=i < size.n_blocks,
                # use_last_activation=i < size.n_blocks
            )
            for i in range(size.n_blocks)
        ])
        self.outc = nn.Conv2d(64, 3, kernel_size=3, padding='same')
        self.outa = nn.Sigmoid()
        # self.outa = nn.ReLU()

        load_weights(self, weights)

    def encode(self, vision):
        return run_sequentially(self.encoder, vision)

    def decode(self, encoded_vision):
        # h = encoded_vision
        # if self.fourier_features is not None:
        #     h = self.fourier_features(h)

        return self.outa(
            self.outc(
                run_sequentially(self.decoder, encoded_vision)
            )
        )

    def forward(self, v):
        return self.decode(self.encode(v))
