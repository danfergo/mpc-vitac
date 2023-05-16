import os
from os import path

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset, predict_batch
from experiments.losses.perceptual_loss import VGGPerceptualLoss
from experiments.nn.vision_cortex import VisionCortex

from experiments.shared.loaders import loader
from experiments.shared.transform import transform

from piqa import SSIM, HaarPSI, VSI, PSNR

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {
    'description': """
        # train visual cortex
    """,
    'config': {
        'lr': 0.1,

        # data
        'data_path': path.join(__location__, '../data/'),
        # 'dataset_name': 'pick_and_place',
        'dataset_name': 'pushing_tower',
        # 'dataset_name': 'sliding_rope',
        'img_size': (112, 112),
        '{data_loader}': lambda: loader('train', inputs=['c:0'], outputs=['c:0']),
        'train_ic_transform': transform(),

        # network
        '{model}': lambda: VisionCortex(
            # weights=e.ws('outputs', 'train_ae', 'runs', '2023-03-07 21:53:47', 'out', 'latest_model'),
            # skip_predictive_model=True  # when True, set the input-output data to the same state.
        ),

        # train
        'loss': VGGPerceptualLoss().to('cuda'),
        '{optimizer}': lambda: optim.Adadelta(e.model.parameters(), lr=e.lr),
        'epochs': 1000,
        'batch_size': 32,
        'batches_per_epoch': 30,
        'feed_size': 16,
        'train_device': 'cuda',

        # validation
        '{metrics}': lambda: [
            SSIM().to(e.train_device),
            PSNR().to(e.train_device),
            HaarPSI().to(e.train_device),
            VSI().to(e.train_device)
        ],
        'metrics_names': [
            'SSIM',
            'PSNR',
            'HaarPSI',
            'VSI'
        ],
        'n_val_batches': 4,
        'val_feed_size': 32,
    }
}

run(
    **config,
    entry=fit_to_dataset,
    listeners=lambda: [
        Validator(),
        ModelSaver(),
        Logger(),
        Plotter(),
        # EBoard(),
        TrainingSamples(loaders=[
            ('train', e.data_loader, 3),
        ])
    ],
    open_e=False,
    src='experiments'
)
