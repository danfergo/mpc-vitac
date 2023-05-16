from torch import nn
import torch

from functools import reduce


class LinearBNR(nn.Module):

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

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DownConv, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        torch.nn.init.xavier_uniform_(self.conv2d.weight)
        self.conv2d.bias.data.fill_(0.0)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv2d(x)
        h = self.batch_norm(h)
        h = self.relu(h)
        return h


class DownBlock(nn.Module):

    def __init__(self, n_layers, in_channels, out_channels=64, kernel_size=3, down_sample=True):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(*[
            DownConv(in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=kernel_size)
            for i in range(n_layers)
        ])
        self.down_sample = nn.MaxPool2d(2) if down_sample else None

    def forward(self, x):
        h = self.layers(x)
        if self.down_sample is not None:
            h = self.down_sample(h)
        return h


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpConv, self).__init__()

        self.conv2dTranspose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv2dTranspose.weight)
        self.conv2dTranspose.bias.data.fill_(0.0)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv2dTranspose(x)
        h = self.batch_norm(h)
        h = self.relu(h)
        return h


class UpBlock(nn.Module):

    def __init__(self, n_layers, in_channels, out_channels=64, kernel_size=3):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(*[
            UpConv(in_channels if i == 0 else out_channels, out_channels, kernel_size)
            for i in range(n_layers)
        ])
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        h = self.layers(x)
        h = self.up_sample(h)
        return h


class DfgPM(nn.Module):
    """
        Dynamics model ot,at -> ot+1.
    """

    def __init__(self, image_size=(3, 224, 224), action_size=5, ae_size=(3, 3, 64), dm_size=512, weights=None):
        super(DfgPM, self).__init__()
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
            in_channels=ae_size[2] + self.action_size,
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

    def forward(self, ot, at):
        h = self.inc(ot)

        h, hs = reduce(lambda ht, l: (l[1](ht[0]), ht[1] + [ht[0]]), enumerate(self.encoder), (h, []))
        hs = hs + [h]
        # print('--->', h.size())

        # predictive model with conv layers.
        at_v_1 = at[:, :, None, None]
        at_v = torch.tile(at_v_1, (1, 1, self.h_size[0], self.h_size[1]))
        h_merged = torch.cat((h, at_v), 1)
        h_next = self.cmlp(h_merged)

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

# def predictive_model(channels=1, grasp_cmd_size=3, img_size=28):
#     # ENCODER
#     i_inp = Input((img_size, img_size, channels), name="grasp")
#
#     g_inp = Input(grasp_cmd_size, name="It")
#     g = dense_norm_relu(64)(g_inp)
#
#     e = conv2D_norm_relu(32, (3, 3), padding='same')(i_inp)
#     e = conv2D_norm_relu(64, (3, 3), padding='same')(e)
#     e = conv2D_norm_relu(64, (3, 3), padding='same')(e)
#     e = MaxPooling2D((2, 2))(e)
#
#     e = conv2D_norm_relu(64, (3, 3), padding='same')(e)
#     e = conv2D_norm_relu(64, (3, 3), padding='same')(e)
#     e = conv2D_norm_relu(64, (3, 3), padding='same')(e)
#     e = mp = MaxPooling2D((2, 2))(e)
#
#     e = Flatten()(e)
#
#     md = Concatenate(axis=1)([g, e])
#
#     md = dense_norm_relu(1024)(md)
#     md = dense_norm_relu(1024)(md)
#     md = dense_norm_relu(1024)(md)
#
#     s_size = mp.shape[1]
#     filters = 64
#
#     md = dense_norm_relu(s_size * s_size * filters)(md)
#
#     # DECODER
#     d = Reshape((s_size, s_size, filters))(md)
#     d = conv2DTranspose_norm_relu(64, (3, 3), strides=1, padding='same')(d)
#     d = conv2DTranspose_norm_relu(64, (3, 3), strides=1, padding='same')(d)
#     d = conv2DTranspose_norm_relu(64, (3, 3), strides=1, padding='same')(d)
#     d = UpSampling2D((2, 2))(d)
#
#     d = conv2DTranspose_norm_relu(64, (3, 3), strides=1, padding='same')(d)
#     d = conv2DTranspose_norm_relu(32, (3, 3), strides=1, padding='same')(d)
#     d = conv2DTranspose_norm_relu(32, (3, 3), strides=1, padding='same')(d)
#     d = UpSampling2D((2, 2))(d)
#
#     decoded = Conv2D(channels, (3, 3), padding='same', activation='sigmoid')(d)
#     # decoded = ReLU()(decoded)
#
#     return Model(inputs=[i_inp, g_inp], outputs=decoded)
