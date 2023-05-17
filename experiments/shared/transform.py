import imgaug.augmenters as iaa

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW

import numpy as np
import cv2

from dfgiatk.experimenter import e


def img_transform(partition, endpoint, modality):
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


def transform():
    seq = iaa.Sequential(
        iaa.OneOf([
            iaa.Sometimes(0.25, iaa.GammaContrast((0.5, 2.0))),
            iaa.Sometimes(0.25, iaa.MotionBlur()),
            # iaa.Affine(rotate=45),
            iaa.AdditiveGaussianNoise(scale=0.01),
            iaa.Add(0.05, per_channel=True),
            iaa.Sharpen(alpha=0.1)
        ])
    )

    # [
    #     iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
    # ] +
    # ([
    # iaa.Sequential([
    #     iaa.Sometimes(0.25, iaa.GammaContrast((0.5, 2.0))),
    #     iaa.Sometimes(0.25, iaa.MotionBlur()),
    #     iaa.Sometimes(0.25, iaa.Cartoon()),
    #     iaa.AdditiveGaussianNoise(scale=(0, 0.2)),
    #     iaa.Sometimes(0.2, iaa.JpegCompression(compression=(0, 50))),
    #     iaa.Sometimes(0.1, iaa.imgcorruptlike.Snow(severity=2)),
    #     # iaa.Sometimes(0.5, iaa.Brightness(severity=(1, 3)))
    # ])
    # iaa.Sequential([
    #     iaa.Sometimes(0.25, iaa.GammaContrast((0.5, 2.0))),
    #     iaa.Sometimes(0.25, iaa.MotionBlur()),
    #     iaa.Sometimes(0.25, iaa.Cartoon()),
    #     iaa.AdditiveGaussianNoise(scale=(0, 0.2)),
    #     iaa.Sometimes(0.2, iaa.JpegCompression(compression=(0, 50))),
    #     # iaa.Sometimes(0.1, iaa.imgcorruptlike.Snow(severity=2)),
    #     # iaa.Sometimes(0.5, iaa.Brightness(severity=(1, 3)))
    # ])
    # ] if image_aug else [])

    def _(batch, samples):
        batch = seq(images=batch)

        return batch

    return _
