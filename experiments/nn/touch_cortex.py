from torch import nn
import torch

from collections import namedtuple

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import run_sequentially, load_weights

TouchCortexSize = namedtuple('TouchCortexSize', ['n_blocks', 'block_size', 'channels'])


class TouchCortex(nn.Module):

    def __init__(self, size: TouchCortexSize = TouchCortexSize(3, 3, 64), weights=None, use_fourier_features=True):
        super(TouchCortex, self).__init__()

        channels = size.channels

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=size.block_size,
                in_channels=3 if i == 0 else channels,
                out_channels=channels
            )
            for i in range(size.n_blocks)
        ])

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=size.block_size,
                in_channels=channels,
                out_channels=channels  # 3 if (i == size.n_blocks) - 1 else size.channels
            )
            for i in range(size.n_blocks)
        ])
        self.outc = nn.Conv2d(64, 3, kernel_size=3, padding='same')
        self.outa = nn.Sigmoid()

        load_weights(self, weights)

    def encode(self, touch):
        return run_sequentially(self.encoder, touch)

    def decode(self, encoded_touch):
        return self.outa(
            self.outc(
                run_sequentially(self.decoder, encoded_touch)
            )
        )

    def forward(self, touch):
        return self.decode(self.encode(touch))
