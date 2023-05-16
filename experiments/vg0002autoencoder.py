import os
from os import path

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

import torch

from experiments.nn.ae_simple import AutoEncoderSimple
from experiments.nn.restnet_ae import ResNetAutoEncoder, get_configs
from experiments.shared.utils import load_dict


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


from dfgiatk.experimenter.event_listeners.training_samples import TrainingSamples
from dfgiatk.train import fit_to_dataset, predict_batch
from experiments.losses.perceptual_loss import VGGPerceptualLoss
from experiments.nn.simpler_brain import SimplerBrain
from experiments.nn.temporal_cortex import TemporalCortex
from experiments.nn.touch_cortex import TouchCortex
from experiments.nn.vision_cortex import VisionCortex

from experiments.shared.visgel_loaders import loader
from experiments.shared.transform import transform

from piqa import SSIM, HaarPSI, VSI, PSNR, LPIPS

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

config = {
    'description': """
        # Train VisGel Auto-encoder
    """,
    'config': {
        'lr': 0.001,

        # data
        'data_path': '/media/danfergo/SSD2T/VisGel/data/',
        'img_size': (128, 128),
        '{data_loader}': lambda: loader('train',
                                        inputs=['c:0', 'l:0'],
                                        outputs=['c:0', 'l:0']
                                        ),
        '{val_loader}': lambda: loader('val',
                                       inputs=['c:0', 'l:0'],
                                       outputs=['c:0', 'l:0']
                                       ),
        # 'train_ic_transform': transform(),

        # network
        '{model}': lambda: AutoEncoderSimple(
            load_dict(e.ws('experiments', 'nn', 'objects365-resnet50.pth'),
                      ResNetAutoEncoder(*get_configs('resnet50'))),  # , skip_connection=True
            load_dict(e.ws('experiments', 'nn', 'objects365-resnet50.pth'),
                      ResNetAutoEncoder(*get_configs('resnet50')))  # , skip_connection=True
            # TemporalCortex()
            # weights=e.ws('outputs', 'train_ae', 'runs', '2023-03-07 21:53:47', 'out', 'latest_model'),
            # skip_predictive_model=True  # when True, set the input-output data to the same state.
        ),

        # train
        'train_device': 'cuda',
        '{perceptual_loss}': lambda: VGGPerceptualLoss().to(e.train_device),
        '{loss}': lambda: (e.perceptual_loss, e.perceptual_loss),
        '{optimizer}': lambda: optim.Adadelta(e.model.parameters(), lr=e.lr),
        'epochs': 100,
        'batch_size': 24,
        'batches_per_epoch': 50,
        'feed_size': 4,

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
        'n_val_batches': 3,
        'val_feed_size': 24,
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
