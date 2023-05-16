from collections import namedtuple

from torch import nn

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import load_weights, run_sequentially

from experiments.nn.utils.fourier_features import FourierFeatures, FourierFeatures2D

TemporalCortexSize = namedtuple('TemporalCortexSize', ['n_blocks', 'block_size', 'channels'])


class TemporalCortex(nn.Module):

    def __init__(self,
                 size: TemporalCortexSize = TemporalCortexSize(2, 3, 64),
                 weights=None,
                 ):
        super(TemporalCortex, self).__init__()

        channels = size.channels

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=size.block_size,
                in_channels=4103 if i == 0 else channels,
                hidden_channels=channels,
                out_channels=channels,
                down_sample=False
            )
            for i in range(size.n_blocks)
        ])

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=size.block_size,
                in_channels=channels,
                hidden_channels=channels,
                out_channels=2048 + 2048 if i == size.n_blocks - 1 else channels,
                up_sample=False,
            )
            for i in range(size.n_blocks)
        ])
        # self.outa = nn.Sigmoid()
        self.outa = nn.ReLU()

        load_weights(self, weights)

    def encode(self, state):
        return run_sequentially(self.encoder, state)

    def decode(self, encoded_state):
        return self.outa(
            run_sequentially(self.decoder, encoded_state)
        )

    def forward(self, state):
        return self.decode(self.encode(state))
