from abc import ABC, abstractmethod

import os
import random

import torch

from torchvision.datasets.folder import make_dataset
from os import path
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dfgiatk.ops.img import denormalize, cvt_batch, CVT_HWC2CHW

CACHE = {}


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi", ".webm")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)


class Labeler:

    def __init__(self, transform=None):
        self.transform = transform

    @abstractmethod
    def get_label(self, s, s_bin):
        pass


class ClassificationLabeler(Labeler):

    def __init__(self, samples=None, transform=None, one_hot=False):
        super().__init__(transform=transform)

        if samples is not None:
            self.classes = list({self.get_class(s): 1 for s in samples}.keys())  # creating a set, unique keys ??
            self.classes.sort()

        self.one_hot = one_hot

    def get_class(self, s):
        return path.basename(path.dirname(s))

    def get_label(self, s, s_bin):
        idx = s_bin if s_bin is not None else self.classes.index(self.get_class(s))
        if self.one_hot:
            _1_hot = np.zeros((len(self.classes, )))
            _1_hot[idx] = 1
            return _1_hot
        return np.array(idx)


class LocalizationLabeler(Labeler):

    def __init__(self, transform=None, locations_path=None):
        super().__init__(transform=transform)
        self.locations = yaml.full_load(open(locations_path)) \
            if locations_path is not None else None

    def get_label(self, s, s_bin):
        filename_w_extension = path.basename(s)
        file_name = filename_w_extension[: filename_w_extension.index('.')]
        folder = path.basename(path.dirname(s))
        if self.locations:
            return np.array([float(c) for c in self.locations[f'{folder}/{file_name}']], dtype=np.float32)
        return np.array([float(c) for c in file_name.split('_')], dtype=np.float32)


class NumpyMapsLabeler(Labeler):

    def __init__(self, arr_path=None, transform=None):
        super().__init__(transform=transform)
        self.arr_path = arr_path

    def get_label(self, s, s_bin):
        arr_path = s if self.arr_path is None else self.arr_path(s)
        if not arr_path in CACHE:
            # if isinstance(s, str):
            #     filename_w_extension = path.basename(s)
            #     file_name = filename_w_extension[: filename_w_extension.index('.')]
            #     folder = path.basename(path.dirname(s))
            #     CACHE[s] = np.load(path.join(self.base_path, folder, file_name + '.npy'))
            # else:
            CACHE[arr_path] = np.load(arr_path)
        return CACHE[arr_path]


class ImageLoader(Labeler):

    def __init__(self, transform=None, img_path=None, use_cache=True, stack=False):
        self.img_path = img_path
        self.use_cache = use_cache
        self.stack = stack
        super().__init__(transform=transform)

    def _get_img(self, s, s_bin, offset):
        img_path = s if self.img_path is None else self.img_path(s, offset)

        # if not self.use_cache or (self.use_cache and img_path not in CACHE):
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            # if not self.use_cache:
            #     return img
            # CACHE[img_path] = img
        #
        # return CACHE[img_path]

    def get_label(self, s, s_bin):
        if self.stack is False:
            return self._get_img(s, s_bin, 0)
        else:
            return [self._get_img(s, s_bin, i) for i in range(self.stack)]


class VideoFrameLoader(Labeler):

    def __init__(self, transform=None, frame_path=None, use_cache=True, stack=False):
        self.frame_path = frame_path
        self.use_cache = use_cache
        self.caps_cache = {}
        self.stack = stack
        super().__init__(transform=transform)

    def _get_img(self, cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, frame = cap.read()
        if not res:
            raise FileNotFoundError()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def get_label(self, s, s_bin):
        video_path, frame_idx = self.frame_path(s)

        cap = cv2.VideoCapture(video_path)

        if self.stack is False:
            ret = self._get_img(cap, frame_idx)
        else:
            ret = [self._get_img(cap, frame_idx + o) for o in range(self.stack)]

        cap.release()
        return ret


class LambdaLoader(Labeler):

    def __init__(self, fn=None, transform=None):
        self.fn = fn
        super().__init__(transform=transform)

    def get_label(self, s, s_bin):
        return self.fn(s, s_bin)


class DatasetSampler(torch.utils.data.IterableDataset):
    def __init__(self,
                 samples,
                 loader=None,
                 loaders=None,
                 labeler=None,
                 labelers=None,
                 epoch_size=1,
                 batch_size=32,
                 random_sampling=True,
                 return_names=False,
                 transform_key=None,
                 self_reset_cache=False,
                 device='cuda'):
        super(DatasetSampler).__init__()

        self.samples = samples

        self.bins = list(range(len(self.samples))) if \
            isinstance(self.samples, list) \
            else None

        self.batch_size = batch_size or len(samples)
        self.epoch_size = epoch_size or len(samples)

        self.loader = loader if not isinstance(loader, list) else None
        self.loaders = loader if loaders is None and isinstance(loader, list) else loaders

        self.labeler = labeler if not isinstance(labeler, list) else None
        self.labelers = labeler if labelers is None and isinstance(labeler, list) else labelers

        self.random_sampling = random_sampling
        self.return_names = return_names
        self.transform_key = transform_key
        self.device = device
        self.self_reset_cache = self_reset_cache

    def get_sample(self, i):
        while True:
            try:
                # Get random sample
                if self.bins is not None:
                    sampled_bin = random.choice(self.bins)
                    samples = self.samples[sampled_bin]
                else:
                    sampled_bin = None
                    samples = self.samples

                    sample_key = random.choice(samples) if self.random_sampling else samples[i]
                    if self.transform_key is not None:
                        sample_key = self.transform_key(sample_key)

                def get(endpoint):
                    return endpoint.get_label(sample_key, sampled_bin)

                return get(self.loader) if self.loaders is None else [get(l) for l in self.loaders], \
                       get(self.labeler) if self.labelers is None else [get(l) for l in self.labelers], \
                       sample_key
            except FileNotFoundError as e:
                pass

    def sample_batch(self):
        if self.self_reset_cache:
            global CACHE
            CACHE = {}

        xs, ys, samples = list(zip(*[self.get_sample(i) for i in range(self.it, self.it + self.batch_size)]))

        if self.loaders is not None:
            xs = list(zip(*xs))
            ys = list(zip(*ys))

        def prepare(endpoint, a):

            as_np = np.array(a)

            if endpoint.transform is not None:
                as_np = endpoint.transform(as_np, samples=samples)

            as_torch = torch.from_numpy(as_np)

            return as_torch.to(self.device)

        xs = prepare(self.loader, xs) if self.loaders is None else \
            tuple([prepare(self.loaders[i], xs[i]) for i in range(len(xs))])

        ys = prepare(self.labeler, ys) if self.labelers is None else \
            tuple([prepare(self.labelers[i], ys[i]) for i in range(len(ys))])

        return (xs, ys) + (samples if self.return_names else tuple())

    def __iter__(self):
        self.it = 0
        return self

    def __next__(self):
        if self.it >= self.epoch_size:
            raise StopIteration
        else:
            b = self.sample_batch()
            self.it += 1
            return b

    @staticmethod
    def load_from_yaml(yaml_path, prepend_path=None):
        return [(path.join(prepend_path, s) if prepend_path is not None else s)
                for s in yaml.full_load(open(yaml_path))]

    @staticmethod
    def load_from_numpy(np_path, prepend_path=None):
        return [(path.join(prepend_path, s) if prepend_path is not None else s)
                for s in np.load(np_path)]


def test():
    import imgaug.augmenters as iaa

    set = 'real_rgb'
    split_file = 'train_split.yaml'
    base_path = '/home/danfergo/Projects/PhD/geltip_simulation/geltip_dataset/dataset/'
    base = path.join('', set)

    samples = DatasetSampler.load_from_yaml(
        path.join(base_path, split_file),
        path.join(base_path, set)
    )

    # labeler = ClassificationLabeler(samples)
    # labeler = LocalizationLabeler()
    labeler = NumpyMapsLabeler(path.join(base_path, 'sim_depth_aligned'))

    def data_preparation(xs):
        xs = (cvt_batch(xs, CVT_HWC2CHW) / 255.0).astype(np.float32)
        return iaa.Sequential([
            iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
            iaa.OneOf([
                iaa.Affine(rotate=0.1),
                iaa.AdditiveGaussianNoise(scale=0.7),
                iaa.Add(50, per_channel=True),
                iaa.Sharpen(alpha=0.5)
            ])
        ])(images=xs)

    loader = DatasetSampler(
        samples,
        labeler=labeler,
        transform=data_preparation
    )

    for x, y in loader:
        imgs = x.detach().cpu().numpy()
        imgs = np.swapaxes(imgs, 1, 2)
        imgs = np.swapaxes(imgs, 2, 3)

        for i in range(x.size()[0]):
            img = denormalize(imgs[i])
            plt.imshow(img)
            plt.show()

    # print(y.size())
    # batch_size=32
    # data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}
    # print(batch[0].size())
    # print(batch)
    # for i in range(len(batch['path'])):
    # data['video'].append(batch['path'][i])
    # data['start'].append(batch['start'][i].item())
    # data['end'].append(batch['end'][i].item())
    # data['tensorsize'].append(batch['video'][i].size())


if __name__ == '__main__':
    test()
