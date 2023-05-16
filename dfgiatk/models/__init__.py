from functools import partial

import torch
from torch import nn

# fix some validation error, while fetching the pretrained models
# https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

