import torch

from torch import nn
from torchvision.models import resnet50


def resnet50(n_activations=2, weights=None):
    """
       Image classification, ResNet-50
    """
    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2", skip_validation=True)

    # override last layer to fit the given prediction task
    model.fc = nn.Linear(in_features=2048, out_features=n_activations, bias=True)

    if weights is not None:
        print('laded weights', weights)
        model.load_state_dict(torch.load(weights))

    return model
