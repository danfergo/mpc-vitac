import torch

from torch import nn
from .resnet import resnet50


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


class resnet_grasp(nn.Module):

    def __init__(self, grasp_size=5, linear_layers_size=512, weights=None):
        super(resnet_grasp, self).__init__()

        self.conv_layers = resnet50(n_activations=512)

        self.linear1 = LinearBNR(in_features=linear_layers_size + grasp_size, out_features=linear_layers_size)
        self.linear2 = LinearBNR(in_features=linear_layers_size, out_features=2)
        self.act = nn.Softmax()

        if weights is not None:
            print('laded weights', weights)
            self.load_state_dict(torch.load(weights))

    def forward(self, ot, at):
        h = self.conv_layers(ot)
        h = torch.cat((h, at), 1)
        h = self.linear1(h)
        h = self.linear2(h)
        return self.act(h)
