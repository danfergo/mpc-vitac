from collections import namedtuple

import torch
from torch import nn

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import load_weights, run_sequentially

AssociativeCortexSize = namedtuple('AssociativeCortexSize', ['n_blocks', 'block_size', 'channels'])


class AssociativeCortex(nn.Module):

    def __init__(self,
                 size: AssociativeCortexSize = AssociativeCortexSize(2, 3, 64),
                 weights=None,
                 ):
        super(AssociativeCortex, self).__init__()

        channels = size.channels
        n_blocks = size.n_blocks

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=size.block_size,
                in_channels=2 * channels if i == 0 else channels,
                hidden_channels=channels,
                out_channels=channels,
                down_sample=False
            )
            for i in range(n_blocks)
        ])

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=size.block_size,
                in_channels=channels,
                hidden_channels=channels,
                out_channels=2 * channels if i == n_blocks - 1 else channels,
                up_sample=False,
            )
            for i in range(n_blocks)
        ])
        # self.outa = nn.Sigmoid()
        self.outa = nn.ReLU()

        load_weights(self, weights)

    def encode(self, vision, touch):
        state = torch.cat((vision, touch), dim=1)
        return run_sequentially(self.encoder, state)

    def decode(self, encoded_state):
        state = self.outa(
            run_sequentially(self.decoder, encoded_state)
        )
        m = state.shape[1] // 2  # middle point, in the channels dimension
        return state[:, :m, :, :], state[:, m:, :, :]

    def forward(self, vision, touch):
        return self.decode(self.encode(vision, touch))
