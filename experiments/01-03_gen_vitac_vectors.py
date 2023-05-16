import os
from os import path
import numpy as np

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset, predict_batch
from experiments.nn.touch_cortex import TouchCortex
from experiments.nn.vision_cortex import VisionCortex

from experiments.shared.loaders import loader
from experiments.shared.transform import transform

# from piqa import SSIM, HaarPSI, VSI, PSNR

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {
    'description': """
        # train touch cortex
    """,
    'config': {
        'lr': 0.1,

        # data
        'data_path': path.join(__location__, '../data/'),
        # 'dataset_name': 'pick_and_place',
        'dataset_name': 'pushing_tower',
        # 'dataset_name': 'sliding_rope',
        'img_size': (112, 112),
        '{data_loader}': lambda: loader('train', inputs=['l:0', 'c:0'], outputs=['l:0', 'c:0']),
        'random_sampling': False,
        'return_names': True,

        # network
        '{vision_cortex}': lambda: VisionCortex(
            weights=e.ws('outputs', '01_train_vision_cortex', 'runs', '2023-04-18 11:29:48', 'out', 'latest_model'),
        ),
        '{touch_cortex}': lambda: TouchCortex(
            weights=e.ws('outputs', '02_train_touch_cortex', 'runs', '2023-04-18 14:43:12', 'out', 'latest_model'),
        ),

        # train
        'batch_size': 1,
        'train_device': 'cpu',

    }
}


def gen_vectors():
    data_loader, vision_cortex, touch_cortex, train_device = e['data_loader', 'vision_cortex', 'touch_cortex', 'train_device']

    vision_cortex.to(train_device)
    touch_cortex.to(train_device)

    # Train some batches
    os.mkdir(e.out('e_vision'))
    os.mkdir(e.out('e_touch'))

    for (vision, touch), _, idx in iter(data_loader):
        e_vision = vision_cortex.encode(vision)
        e_touch = touch_cortex.encode(touch)

        e_vision = e_vision.squeeze().detach().cpu().numpy()
        e_touch = e_touch.squeeze().detach().cpu().numpy()

        # print(e_vision.flatten()[0:3], e_touch.flatten()[0:3])
        np.save(e.out('e_vision', f'{idx}.npy'), e_vision)
        np.save(e.out('e_touch', f'{idx}.npy'), e_touch)


run(
    **config,
    entry=gen_vectors,
    open_e=False,
    src='experiments'
)
