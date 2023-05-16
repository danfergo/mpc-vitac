from functools import reduce
import torch


def load_weights(self, weights):
    if weights is not None:
        print('loaded weights', weights)
        self.load_state_dict(torch.load(weights))


def run_sequentially(module_list, inputs):
    return reduce(
        lambda ht, it_layers: (it_layers[1](ht)),
        enumerate(module_list),
        inputs  # initial values (inputs and empty list to collect Hs)
    )
