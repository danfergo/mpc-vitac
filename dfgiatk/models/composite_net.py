import torch

from torch import nn

from dfgiatk.models.dfg_pm import DfgViTacPM
from dfgiatk.models.resnet import resnet50


class CompositeNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CompositeNet, self).__init__()

        self.predictive_model = DfgViTacPM(*args, **kwargs)
        self.grasp_success = resnet50()
        # self.act = nn.Softmax()

    def forward(self, i_b, g):
        i_d = self.predictive_model(i_b, g)
        s = self.grasp_success(i_d)
        return s
        # print(s.size())
        # return self.act(s)
