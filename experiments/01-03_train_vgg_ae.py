import os
import sys
from os import path

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver
from torch import nn, optim

import torch

from experiments.nn.vgg_ae import VGGAutoEncoder, get_configs


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

from experiments.shared.loaders import loader
from experiments.shared.transform import transform

from piqa import SSIM, HaarPSI, VSI, PSNR, LPIPS

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        # model_dict = model.state_dict()
        # print(checkpoint)
        # model_dict.update(checkpoint['state_dict'])
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model


config = {
    'description': """
        # train touch cortex
    """,
    'config': {
        'lr': 0.00002,

        # data
        'data_path': path.join(__location__, '../data/'),
        'dataset_name': 'pick_and_place',
        # 'dataset_name': 'pushing_tower',
        # 'dataset_name': 'sliding_rope',
        # 'img_size': (112, 112),
        'img_size': (224, 224),
        '{data_loader}': lambda: loader('train',
                                        inputs=['c:0'],
                                        outputs=['c:0']
                                        ),
        # 'train_ic_transform': transform(),

        # network
        # '{model}': lambda: SimplerBrain(
        #     VisionCortex(),
        #     TouchCortex(),
        #     TemporalCortex()
        #     # weights=e.ws('outputs', 'train_ae', 'runs', '2023-03-07 21:53:47', 'out', 'latest_model'),
        #     # skip_predictive_model=True  # when True, set the input-output data to the same state.
        # ),
        '{model}': lambda: load_dict(e.ws('experiments', 'nn', 'imagenet-vgg16.pth'),
                                     VGGAutoEncoder(get_configs('vgg16'))),
        # train
        'train_device': 'cuda',
        '{perceptual_loss}': lambda: VGGPerceptualLoss().to(e.train_device),
        'complex_loss': False,
        # '{loss}': lambda: (e.perceptual_loss, e.perceptual_loss),
        '{loss}': lambda: e.perceptual_loss,
        '{optimizer}': lambda: optim.Adadelta(e.model.parameters(), lr=e.lr),
        'epochs': 1000,
        'batch_size': 20,
        'batches_per_epoch': 2,
        'feed_size': 10,

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
