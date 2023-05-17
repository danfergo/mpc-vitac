import os
from os import path
import numpy as np

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset, predict_batch
from experiments.nn.touch_cortex import TouchCortex
from experiments.nn.vision_cortex import VisionCortex

from experiments.shared.vitacworld_loaders import loader
from experiments.shared.transform import transform

# from piqa import SSIM, HaarPSI, VSI, PSNR

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

config = {
    'description': """
        # train touch cortex
    """,
    'config': {
        'lr': 0.1,

        # data
        '{data_path}': lambda: e.ws('outputs', '03_gen_vitac_vectors', 'runs', '2023-04-18 21:50:08'),
        # 'data_path': path.join(__location__, '../data/'),
        # 'dataset_name': 'pick_and_place',
        # 'dataset_name': 'pushing_tower',
        'dataset_name': 'out',

        # 'dataset_name': 'sliding_rope',
        'img_size': (112, 112),
        'vectors_size': (64, 14, 14),
        '{data_loader}': lambda: loader('train', inputs=['ec:0', 'el:0'], outputs=['ec:1', 'el:1']),


        # '{data_loader}': lambda: loader('train', inputs=['l:0', 'c:0'], outputs=['l:0', 'c:0']),
        'random_sampling': False,
        'return_names': True,

        # train
        'batch_size': 1,
        'train_device': 'cpu',

    }
}


def vis_vectors():
    data_loader = e['data_loader']

    e_vision_flat = []
    e_touch_flat = []

    for (vision, touch), _, idx in iter(data_loader):
        # e_vision = vision_cortex.encode(vision)
        # e_touch = touch_cortex.encode(touch)

        e_vision = vision.squeeze().detach().cpu().numpy().flatten()
        e_touch = touch.squeeze().detach().cpu().numpy().flatten()

        print(e_touch[0:3], e_vision[0:3])
        e_vision_flat.append(e_vision)
        e_touch_flat.append(e_touch)

        # print(e_vision.shape)
        # np.save(e.out('e_vision', f'{idx}.npy'), e_vision)
        # np.save(e.out('e_touch', f'{idx}.npy'), e_touch)

    pca = PCA(n_components=2)
    e_vision_pca = pca.fit_transform(e_vision_flat)
    e_touch_pca = pca.fit_transform(e_touch_flat)

    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    targets = [0, 1, 2]
    # for target, color in zip(targets, colors):
    # plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], c=color, label=iris.target_names[target])
    plt.scatter(e_vision_pca[:, 0], e_vision_pca[:, 1], color=(0.0, 1.0, 0.0, 0.3), label='e_vision_pca')
    plt.scatter(e_touch_pca[:, 0], e_touch_pca[:, 1], color=(0.0, 0.0, 1.0, 0.3), label='e_touch_pca')

    for i in range(len(e_vision_pca)):
        plt.annotate(str(i), (e_vision_pca[i, 0], e_vision_pca[i, 1]), c='gray')
        plt.annotate(str(i), (e_touch_pca[i, 0], e_touch_pca[i, 1]), c='gray')

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.title('PCA of Iris Dataset')
    plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # from sklearn.decomposition import PCA
    #
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    #
    # plt.figure(figsize=(8, 6))
    # colors = ['r', 'g', 'b']
    # targets = [0, 1, 2]
    # for target, color in zip(targets, colors):
    #     plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], c=color, label=iris.target_names[target])
    #
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.legend()
    # plt.title('PCA of Iris Dataset')
    # plt.show()


run(
    **config,
    entry=vis_vectors,
    open_e=False,
    src='experiments'
)
