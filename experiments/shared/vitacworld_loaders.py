import os
from os import path
import random

from dfgiatk.experimenter import e, run
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import ImageLoader, ClassificationLabeler, LambdaLoader, NumpyMapsLabeler
import numpy as np
import cv2

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW, CVT_CHW2HWC
from experiments.shared.transform import img_transform


def loader(partition, inputs=['c:0'], outputs=['c:1'], stack=False):
    dataset_path = e.data_path  # path.join(e.data_path, e.dataset_name)

    samples = [(s, rec, len(os.listdir(path.join(dataset_path, s, rec, 'c'))))
               for s in ['pick_and_place', 'pushing_tower']
               for rec in os.listdir(path.join(dataset_path, s))]

    def img_path(modality, offset):
        def _(key, rel_offset):
            p = path.join(dataset_path, f'{key[0]}/{key[1]}',
                          modality,
                          f'frame_{str(key[2] + offset + rel_offset).zfill(5)}.jpg')

            if not os.path.exists(p):
                raise FileNotFoundError()
            return p

        return _

    def encoded_vector_path(modality, offset):
        def _(key):
            return path.join(dataset_path, modality, f'{str(key + offset)}.npy')

        return _

    positions = {'pushing_tower': {}, 'pick_and_place': {}}
    for s in samples:
        positions_path = path.join(dataset_path, s[0], s[1], 'p.npy')
        with open(positions_path, 'rb') as f:
            positions[s[0]][s[1]] = np.load(f).reshape(-1, 7).astype(np.float32)

    loaders = {
        'l': lambda offset, endpoint: ImageLoader(
            transform=img_transform(partition, endpoint, 'l'),
            img_path=img_path('l', offset),
            stack=stack if endpoint == 'i' else False
        ),
        'r': lambda offset, endpoint: ImageLoader(
            transform=img_transform(partition, endpoint, 'r'),
            img_path=img_path('r', offset),
            stack=stack if endpoint == 'i' else False
        ),
        'c': lambda offset, endpoint: ImageLoader(
            transform=img_transform(partition, endpoint, 'c'),
            img_path=img_path('c', offset),
            stack=stack if endpoint == 'i' else False
        ),
        'a': lambda offset, endpoint: LambdaLoader(
            lambda key, _: positions[key[0]][key[1]][key[2] + 1] - positions[key[0]][key[1]][key[2]],
        ),
        # 'p': lambda offset, endpoint: LambdaLoader(
        #     lambda key, _: load_positions(key, offset)
        #     # transform=img_transform(endpoint, 'p'),
        #     # img_path=img_path('c', offset)
        # ),

        # 'ec': lambda offset, endpoint: NumpyMapsLabeler(
        #     # transform=img_transform(endpoint, 'c'),
        #     arr_path=encoded_vector_path('e_vision', offset)
        # ),
        # 'el': lambda offset, endpoint: NumpyMapsLabeler(
        #     # transform=img_transform(endpoint, 'c'),
        #     arr_path=encoded_vector_path('e_touch', offset)
        # ),
    }

    def get_loader(k, endpoint):
        key, offset = tuple(k.split(':'))
        return loaders[key](int(offset), endpoint)

    def transform_key(key):
        rec_sample = samples[key]
        rnd_frame = random.randint(0, rec_sample[2] - stack - 2)
        return rec_sample[0], rec_sample[1], rnd_frame

    return DatasetSampler(
        samples=np.array(list(range(len(samples)))),
        transform_key=transform_key,
        loader=get_loader(inputs[0], 'i') if len(inputs) == 1 else [get_loader(inp, 'i') for inp in inputs],
        labeler=get_loader(outputs[0], 'o') if len(outputs) == 1 else [get_loader(out, 'o') for out in outputs],
        epoch_size=None if ('random_sampling' in e and not e['random_sampling']) else
        (e.batches_per_epoch if partition == 'train' else e.n_val_batches),
        batch_size=e.batch_size,
        device=e.train_device,
        random_sampling=e.random_sampling if 'random_sampling' in e else True,
        return_names=e.return_names if 'return_names' in e else False,
        self_reset_cache=True
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
