import os
from os import path

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset, predict_batch
from experiments.nn.associative_cortex import AssociativeCortex
from experiments.nn.temporal_cortex import TemporalCortex
from experiments.nn.vision_cortex import VisionCortex

from experiments.shared.vitacworld_loaders import loader
from experiments.shared.transform import transform

from piqa import SSIM, HaarPSI, VSI, PSNR

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {
    'description': """
        # train associative cortex
    """,
    'config': {
        'lr': 0.1,

        # data
        # 'data_path': path.join(__location__, '../data/'),
        '{data_path}': lambda: e.ws('outputs', '03_gen_vitac_vectors', 'runs', '2023-04-18 21:50:08'),
        # 'dataset_name': 'pick_and_place',
        'dataset_name': 'out',
        # 'dataset_name': 'sliding_rope',
        'vectors_size': (64, 14, 14),
        '{data_loader}': lambda: loader('train', inputs=['ec:0', 'el:0'], outputs=['ec:1', 'el:1']),
        # 'train_ic_transform': transform(),

        # network
        '{model}': lambda: AssociativeCortex(
            # weights=e.ws('outputs', 'train_ae', 'runs', '2023-03-07 21:53:47', 'out', 'latest_model'),
            # skip_predictive_model=True  # when True, set the input-output data to the same state.
        ),

        # train
        'loss': (nn.MSELoss(), nn.MSELoss()),
        '{optimizer}': lambda: optim.Adadelta(e.model.parameters(), lr=e.lr),
        'epochs': 50000,
        'batch_size': 32,
        'batches_per_epoch': 100,
        'feed_size': 32,
        'train_device': 'cuda',

        # validation
        '{metrics}': lambda: [
            # SSIM().to(e.train_device),
            # PSNR().to(e.train_device),
            # HaarPSI().to(e.train_device),
            # VSI().to(e.train_device)
        ],
        'metrics_names': [
            # 'SSIM', 'PSNR', 'HaarPSI', 'VSI'
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
        # TrainingSamples(loaders=[
        #     ('train', e.data_loader, 3),
        # ])
    ],
    open_e=False,
    src='experiments'
)
