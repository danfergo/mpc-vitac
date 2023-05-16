from collections import namedtuple

from torch import nn

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import load_weights, run_sequentially

from experiments.nn.utils.fourier_features import FourierFeatures, FourierFeatures2D

VisionCortexSize = namedtuple('VisualCortexSize', ['n_blocks', 'block_size', 'channels'])


class VisionCortex(nn.Module):

    def __init__(self, size: VisionCortexSize = VisionCortexSize(3, 3, 64), weights=None, train_depths=None):
        super(VisionCortex, self).__init__()

        channels = size.channels

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=size.block_size,
                in_channels=3 if i == 0 else channels,
                hidden_channels=channels,
                out_channels=channels,
                hidden_activation='relu',
                down_sample=True
            )
            for i in range(size.n_blocks)
        ])

        e_slice_tuple = train_depths if train_depths is not None else (0, size.n_blocks)
        self.e_slice = slice(*e_slice_tuple)

        slice_length = e_slice_tuple[1] - e_slice_tuple[0]
        p_start = size.n_blocks - 1 - e_slice_tuple[0]
        self.d_slice = slice(*(p_start, p_start + slice_length))

        # self.fourier_features = None
        # if use_fourier_features:
        #     self.fourier_features = FourierFeatures2D(channels, channels)
        #     channels = 2 * channels

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=size.block_size,
                in_channels=channels,
                hidden_channels=channels,
                out_channels=3 if (i == size.n_blocks - 1) else channels,
                up_sample_before=True,
                hidden_activation='relu',
                out_activation='sigmoid' if i == size.n_blocks - 1 else 'relu',
                up_sample=True
                # up_sample=i < size.n_blocks,
            )
            for i in range(size.n_blocks)
        ])
        # self.outc = nn.Conv2d(64, 3, kernel_size=3, padding='same')
        # self.outa = nn.Sigmoid()
        # self.outa = nn.ReLU()

        load_weights(self, weights)

    def encode(self, vision):
        return run_sequentially(self.encoder[self.e_slice], vision)

    def decode(self, encoded_vision):
        return run_sequentially(self.decoder[self.d_slice], encoded_vision)

    def forward(self, v):
        # 0.29113525

        return self.decode(self.encode(v)) # - 0.41215315
