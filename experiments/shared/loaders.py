import os
from math import sin, cos
from os import path

import yaml

from dfgiatk.experimenter import e, run
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import ImageLoader, ClassificationLabeler, LambdaLoader, NumpyMapsLabeler
import numpy as np
import cv2

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW, CVT_CHW2HWC


def loader(partition, inputs=['c:0'], outputs=['c:1']):
    dataset_path = path.join(e.data_path, e.dataset_name)

    dataset_length = len(os.listdir(path.join(dataset_path, os.listdir(dataset_path)[0])))

    def img_path(modality, offset):
        def _(key):
            return path.join(dataset_path, modality, f'frame_{str(key + offset).zfill(5)}.jpg')

        return _

    def encoded_vector_path(modality, offset):
        def _(key):
            return path.join(dataset_path, modality, f'{str(key + offset)}.npy')

        return _

    def img_transform(endpoint, modality):
        aug_key = f'{partition}_{endpoint}{modality}_transform'
        aug = aug_key in e

        def _(img_batch, samples):
            batch = np.array([
                cv2.resize(im, e.img_size) for im in img_batch
            ])
            if aug:
                batch = e[aug_key](batch, samples)
            return cvt_batch((batch / 255.0), CVT_HWC2CHW).astype(np.float32)

        return _

    with open(dataset_path + '/p.npy', 'rb') as f:
        positions = np.load(f).reshape(-1, 7).astype(np.float32)

    def load_positions(key, offset):
        return positions[key + offset]

    loaders = {
        'l': lambda offset, endpoint: ImageLoader(
            transform=img_transform(endpoint, 'l'),
            img_path=img_path('l', offset)
        ),
        'r': lambda offset, endpoint: ImageLoader(
            transform=img_transform(endpoint, 'r'),
            img_path=img_path('r', offset)
        ),
        'c': lambda offset, endpoint: ImageLoader(
            transform=img_transform(endpoint, 'c'),
            img_path=img_path('c', offset)
        ),
        # 'p': lambda offset, endpoint: LambdaLoader(
        #     lambda key, _: load_positions(key, offset)
        #     # transform=img_transform(endpoint, 'p'),
        #     # img_path=img_path('c', offset)
        # ),
        'a': lambda offset, endpoint: LambdaLoader(
            lambda key, _: load_positions(key, offset + 1) - load_positions(key, offset)
        ),
        'ec': lambda offset, endpoint: NumpyMapsLabeler(
            # transform=img_transform(endpoint, 'c'),
            arr_path=encoded_vector_path('e_vision', offset)
        ),
        'el': lambda offset, endpoint: NumpyMapsLabeler(
            # transform=img_transform(endpoint, 'c'),
            arr_path=encoded_vector_path('e_touch', offset)
        ),
    }

    def get_loader(k, endpoint):
        key, offset = tuple(k.split(':'))
        return loaders[key](int(offset), endpoint)

    return DatasetSampler(
        samples=np.array(list(range(dataset_length - 1))),
        loader=get_loader(inputs[0], 'i') if len(inputs) == 1 else [get_loader(inp, 'i') for inp in inputs],
        labeler=get_loader(outputs[0], 'o') if len(outputs) == 1 else [get_loader(out, 'o') for out in outputs],
        epoch_size=None if ('random_sampling' in e and not e['random_sampling']) else
        (e.batches_per_epoch if partition == 'train' else e.n_val_batches),
        batch_size=e.batch_size,
        transform_key=None,
        device=e.train_device,
        random_sampling=e.random_sampling if 'random_sampling' in e else True,
        return_names=e.return_names if 'return_names' in e else False
    )


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


    def test_loader():
        loader_ = loader('train')
        for inputs, outputs in loader_:
            x_np = cvt_batch(inputs.cpu().numpy(), CVT_CHW2HWC)
            y_np = cvt_batch(outputs.cpu().numpy(), CVT_CHW2HWC)
            n_samples = x_np.shape[0]
            for i in range(n_samples):
                cv2.imshow('frames', np.concatenate([
                    cv2.cvtColor(x_np[i], cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(y_np[i], cv2.COLOR_RGB2BGR)
                ], axis=1))
                cv2.waitKey(-1)


    run(
        description='Just testing the loaders',
        config={
            'batch_size': 32,
            'batches_per_epoch': 10,
            'train_device': 'cuda',
            'data_path': path.join(__location__, '../../data/'),
            'dataset_name': 'pushing_tower',
        },
        tmp=True,
        entry=test_loader
    )
