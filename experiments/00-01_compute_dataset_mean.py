import os
from os import path
import numpy as np

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from experiments.shared.loaders import loader

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {
    'description': """
        # train touch cortex
    """,
    'config': {
        # data
        'data_path': path.join(__location__, '../data/'),
        # 'dataset_name': 'pick_and_place',
        'dataset_name': 'pushing_tower',
        # 'dataset_name': 'sliding_rope',
        'img_size': (112, 112),
        '{data_loader}': lambda: loader('train', inputs=['l:0', 'c:0'], outputs=['l:0', 'c:0']),
        'random_sampling': False,
        'return_names': False,
        'train_device': 'cpu',
        'batch_size': 1
    }
}


def gen_mean_dataset():
    data_loader = e['data_loader']

    # vision_cortex.to(device)
    # touch_cortex.to(device)
    # # Train some batches
    # os.mkdir(e.out('e_vision'))
    # os.mkdir(e.out('e_touch'))
    v_mean = None
    t_mean = None
    n = 0
    for (vision, touch), idx in iter(data_loader):
        vision = vision.squeeze().detach().cpu().numpy()
        touch = touch.squeeze().detach().cpu().numpy()
        v_mean = vision if v_mean is None else v_mean + vision
        t_mean = touch if t_mean is None else t_mean + touch
        n += 1

    v_mean = np.mean(v_mean / n)
    t_mean = np.mean(t_mean / n)

    # 0.41215315
    # 0.29113525

    print(v_mean, t_mean)
    #     e_vision = vision_cortex.encode(vision)
    #     e_touch = touch_cortex.encode(touch)
    #
    #
    #     np.save(e.out('e_vision', f'{idx}.npy'), e_vision)
    #     np.save(e.out('e_touch', f'{idx}.npy'), e_touch)


run(
    **config,
    entry=gen_mean_dataset,
    open_e=False,
    src='experiments',
    tmp=True
)
