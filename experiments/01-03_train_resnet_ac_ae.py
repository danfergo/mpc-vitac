import os
import sys
from os import path

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

import torch

from experiments.nn.restnet_ae import ResNetAutoEncoder, get_configs
from experiments.nn.simpler_brain import SimplerBrain
from experiments.nn.temporal_cortex import TemporalCortex


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset, predict_batch
from experiments.losses.perceptual_loss import VGGPerceptualLoss
# from experiments.nn.simpler_brain import SimplerBrain
# from experiments.nn.temporal_cortex import TemporalCortex
# from experiments.nn.touch_cortex import TouchCortex
# from experiments.nn.vision_cortex import VisionCortex

from experiments.shared.vitacworld_loaders import loader
# from experiments.shared.transform import transform

from piqa import SSIM, HaarPSI, VSI, PSNR, LPIPS

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))





config = {
    'description': """
        # train touch cortex
    """,
    'config': {
        'lr': 0.01,

        # data
        'data_path': path.join(__location__, '../data/'),
        'dataset_name': 'pick_and_place',
        # 'dataset_name': 'pushing_tower',
        # 'dataset_name': 'sliding_rope',
        'img_size': (128, 128),
        # 'img_size': (224, 224),
        '{data_loader}': lambda: loader('train',
                                        inputs=['c:0', 'l:0', 'a:0'],
                                        outputs=['c:1', 'l:1']
                                        ),
        # 'train_ic_transform': transform(),

        # network
        '{model}': lambda: SimplerBrain(
            load_dict(e.ws('experiments', 'nn', 'objects365-resnet50.pth'),
                      ResNetAutoEncoder(*get_configs('resnet50'))),
            load_dict(e.ws('experiments', 'nn', 'objects365-resnet50.pth'),
                      ResNetAutoEncoder(*get_configs('resnet50'))),
            TemporalCortex()
            # VisionCortex(),
            # TouchCortex(),
            # weights=e.ws('outputs', 'train_ae', 'runs', '2023-03-07 21:53:47', 'out', 'latest_model'),
            # skip_predictive_model=True  # when True, set the input-output data to the same state.
        ),
        # '{model}': lambda: load_dict(e.ws('experiments', 'nn', 'objects365-resnet50.pth'),
        #                              ResNetAutoEncoder(*get_configs('resnet50'))),
        # train
        'train_device': 'cuda',
        '{perceptual_loss}': lambda: VGGPerceptualLoss().to(e.train_device),
        'complex_loss': False,
        '{loss}': lambda: (e.perceptual_loss, e.perceptual_loss),
        # '{loss}': lambda: e.perceptual_loss,
        '{optimizer}': lambda: optim.Adadelta(e.model.parameters(), lr=e.lr),
        'epochs': 1000,
        'batch_size': 32,
        'batches_per_epoch': 50,
        'feed_size': 2,

        # validation
        '{metrics}': lambda: [
            SSIM().to(e.train_device),
            PSNR().to(e.train_device),
            LPIPS().to(e.train_device)
            # HaarPSI().to(e.train_device),
            # VSI().to(e.train_device)
        ],
        # 'HaarPSI', 'VSI'
        'metrics_names': ['SSIM', 'PSNR', 'LPIPS'],
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
