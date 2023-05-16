import os
# from math import sin, cos
from os import path
import random

import yaml

from dfgiatk.experimenter import e, run
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import ImageLoader, ClassificationLabeler, LambdaLoader, NumpyMapsLabeler
import numpy as np
import cv2

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW, CVT_CHW2HWC


def loader(partition, inputs=['c:0'], outputs=['c:1'], input_frames=1):
    data_part = 'data_seen' if partition == 'train' else 'data_unseen'

    dataset_path = path.join(e.data_path, data_part, 'images')
    files_path = path.join(e.data_path, data_part, 'files.yaml')
    samples = yaml.load(open(files_path, 'r'), yaml.FullLoader)

    # dataset_length = len(os.listdir(path.join(dataset_path, os.listdir(dataset_path)[0])))

    def img_path(modality, offset):
        def _(key, rel_offset):
            p = path.join(dataset_path, modality, key[0], key[1],
                          f'frame{str(key[3] + offset + rel_offset).zfill(4)}.jpg')
            if not os.path.exists(p):
                raise FileNotFoundError()
            return p

        return _

    def encoded_vector_path(modality, offset):
        def _(key):
            return path.join(dataset_path, modality, f'{str(key + offset)}.npy')

        return _

    def img_transform(endpoint, modality):
        aug_key = f'{partition}_{endpoint}{modality}_transform'
        aug = aug_key in e

        def _(img_batch, samples):
            if len(img_batch[0].shape) == 3:
                batch = np.array([
                    cv2.resize(im, e.img_size) for im in img_batch
                ])
            else:
                batch = np.array([
                    [cv2.resize(im, e.img_size) for im in stack] for stack in img_batch
                ])
            if aug:
                batch = e[aug_key](batch, samples)
            return cvt_batch((batch / 255.0), CVT_HWC2CHW).astype(np.float32)

        return _

    # with open(dataset_path + '/p.npy', 'rb') as f:
    #     positions = np.load(f).reshape(-1, 7).astype(np.float32)

    # def load_positions(key, offset):
    #     return positions[key + offset]

    loaders = {
        'l': lambda offset, endpoint: ImageLoader(
            transform=img_transform(endpoint, 'l'),
            img_path=img_path('touch', offset),
            stack=input_frames if endpoint == 'i' else 1,
            use_cache=False
        ),
        # 'r': lambda offset, endpoint: ImageLoader(
        #     transform=img_transform(endpoint, 'r'),
        #     img_path=img_path('r', offset)
        # ),
        'c': lambda offset, endpoint: ImageLoader(
            transform=img_transform(endpoint, 'c'),
            img_path=img_path('vision', offset),
            stack=input_frames if endpoint == 'i' else 1,
            use_cache=False
        ),
        # 'p': lambda offset, endpoint: LambdaLoader(
        #     lambda key, _: load_positions(key, offset)
        #     # transform=img_transform(endpoint, 'p'),
        #     # img_path=img_path('c', offset)
        # ),
        # 'a': lambda offset, endpoint: LambdaLoader(
        #     lambda key, _: load_positions(key, offset + 1) - load_positions(key, offset)
        # ),
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

    def transform_key(key):
        rec_sample = samples[key]
        sample_with_rnd_frame = rec_sample + (random.randint(0, rec_sample[2] - input_frames),)
        return sample_with_rnd_frame

    return DatasetSampler(
        samples=np.array(list(range(len(samples)))),
        loader=get_loader(inputs[0], 'i') if len(inputs) == 1 else [get_loader(inp, 'i') for inp in inputs],
        labeler=get_loader(outputs[0], 'o') if len(outputs) == 1 else [get_loader(out, 'o') for out in outputs],
        epoch_size=None if ('random_sampling' in e and not e['random_sampling']) else
        (e.batches_per_epoch if partition == 'train' else e.n_val_batches),
        batch_size=e.batch_size,
        transform_key=transform_key,
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
                print(x_np[i].shape, x_np[i].dtype, np.max(x_np[i]), np.min(x_np[i]))
                frame = np.concatenate([
                    cv2.cvtColor(x_np[i], cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(y_np[i], cv2.COLOR_RGB2BGR)
                ], axis=1)
                if i == 0:
                    cv2.imwrite('frame.jpg', frame * 255 // 1.0)
                # cv2.imshow('frames', frame)
                # cv2.waitKey(-1)


    run(
        description='Just testing the loaders',
        config={
            'batch_size': 32,
            'batches_per_epoch': 10,
            'train_device': 'cuda',
            'img_size': (128, 128),
            'cache_data': False,
            'data_path': '/media/danfergo/SSD2T/VisGel/data/',
        },
        tmp=True,
        entry=test_loader
    )
