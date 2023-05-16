from torch import nn
import torch

from functools import reduce
from experiments.nn.utils.blocks import DownBlock, UpBlock


class DfgVisPM(nn.Module):
    """
        Dynamics model ot,at -> ot+1.
    """

    def __init__(self, image_size=(3, 224, 224), action_size=5, ae_size=(3, 3, 64), dm_size=512, weights=None,
                 skip_predictive_model=False):
        super(DfgVisPM, self).__init__()

        self.skip_predictive_model = skip_predictive_model
        self.ae_size = ae_size
        self.action_size = action_size

        h_size = ((image_size[1]) // (2 ** ae_size[0]),
                  (image_size[2]) // (2 ** ae_size[0]))

        self.h_flat_shape = ae_size[2] * h_size[0] * h_size[1]

        self.h_shape = (ae_size[2], h_size[0], h_size[1])
        self.h_size = h_size

        # downstream
        self.inc = nn.Conv2d(3, 64, kernel_size=3, padding='same')

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=ae_size[1],
                in_channels=ae_size[2]
            )
            for i in range(ae_size[0])
        ])

        # action merging

        # dynamics model
        # self.mlp1 = LinearBNR(self.h_flat_shape + self.action_size, dm_size)
        # self.mlp2 = LinearBNR(dm_size, dm_size)
        # self.mlp3 = LinearBNR(dm_size, self.h_flat_shape)
        self.cmlp = DownBlock(
            n_layers=ae_size[1],
            in_channels=ae_size[2],  # + self.action_size,
            out_channels=ae_size[2],
            down_sample=False
        )

        # upstream
        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=ae_size[1],
                in_channels=ae_size[2],  # * 2 if i == 0 else ae_size[2],
                out_channels=ae_size[2]
            )
            for i in range(ae_size[0])
        ])

        self.fuse = DownBlock(
            n_layers=ae_size[1],
            in_channels=ae_size[2] + image_size[0],
            out_channels=ae_size[2],
            down_sample=False
        )
        self.outc = nn.Conv2d(64, 3, kernel_size=3, padding='same')
        self.outa = nn.Sigmoid()
        # self.outa = nn.ReLU()

        if weights is not None:
            print('loaded weights', weights)
            self.load_state_dict(torch.load(weights))

    def forward(self, ot):
        h = self.inc(ot)

        h, hs = reduce(lambda ht, l: (l[1](ht[0]), ht[1] + [ht[0]]), enumerate(self.encoder), (h, []))
        hs = hs + [h]

        # predictive model with conv layers.
        # at_v_1 = at[:, :, None, None]
        # at_v = torch.tile(at_v_1, (1, 1, self.h_size[0], self.h_size[1]))
        # h_merged = torch.cat((h, at_v), 1)

        if not self.skip_predictive_model:
            h_next = self.cmlp(h)

            # predictive model with linear layers.
            # h_flat = torch.reshape(h, (- 1, self.h_flat_shape))
            # h_merged = torch.cat((h_flat, at), 1)
            # md = self.mlp1(h_merged)
            # md = self.mlp2(md)
            # h_next_flat = self.mlp3(md)
            # h_next = torch.reshape(h_next_flat, (-1, *self.h_shape))

            # h = torch.cat((h, h_next), 1)
            h = h_next

        out = reduce(lambda h_, l: l[1](h_), enumerate(self.decoder), h)

        out = self.fuse(torch.cat((ot, out), 1))

        out = self.outc(out)
        out = self.outa(out)
        return out
