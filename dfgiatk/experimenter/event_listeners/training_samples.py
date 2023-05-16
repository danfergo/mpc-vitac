import os

import cv2
import numpy as np

from ..experimenter import e
from ...ops.img import CVT_CHW2HWC, cvt_batch
from ...train import predict_batch

import torch


class TrainingSamples:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self, loaders):
        self.n_heads = len(e.loss) if isinstance(e.loss, tuple) else 1
        self.loaders = loaders
        os.mkdir(e.out('samples'))

    def on_validation_end(self, ev):
        history = ev['history']

        epoch = history['epoch']
        full_frame = None
        with torch.inference_mode():

            for i, (name, ldr, n_samples) in enumerate(self.loaders):

                b_x, b_y_true = next(iter(ldr))

                batch_loss, b_y_pred = predict_batch((b_x, b_y_true))

                # if isinstance(x, tuple):
                for j in range(self.n_heads):

                    if self.n_heads == 1:
                        x = b_x
                        y_true = b_y_true
                        y_pred = b_y_pred
                    else:
                        x = b_x[j]
                        y_true = b_y_true[j]
                        y_pred = b_y_pred[j]

                    x = x.detach().cpu().numpy()
                    y_true = y_true.detach().cpu().numpy()
                    y_pred = y_pred.detach().cpu().numpy()

                    if len(x.shape) == 5:
                        x = np.array([x[i, 0, :, :, :] for i in range(x.shape[0])])

                    x = cvt_batch(x, CVT_CHW2HWC)
                    y_true = cvt_batch(y_true, CVT_CHW2HWC)
                    y_pred = cvt_batch(y_pred, CVT_CHW2HWC)

                    title_bar = np.ones((60, x[0].shape[1] + y_pred[0].shape[1] + y_true[0].shape[1], 3),
                                        np.uint8) * 255
                    title_bar = cv2.putText(title_bar, name,
                                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                                            cv2.LINE_AA)

                    if i == 0:
                        title_bar = cv2.putText(title_bar, 'epoch: ' + str(epoch + 1),
                                                (y_pred[0].shape[1] + y_true[0].shape[1] - 100, 40),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                    frame = np.concatenate([title_bar] + [
                        np.concatenate([
                            cv2.cvtColor((x[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB),
                            cv2.cvtColor((y_true[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB),
                            cv2.cvtColor((y_pred[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB),
                        ], axis=1)
                        for i in range(n_samples)], axis=0)

                    full_frame = frame if full_frame is None else np.concatenate([
                        full_frame,
                        frame
                    ], axis=0)

        cv2.imwrite(e.out(f'samples/{epoch + 1}.jpg'), full_frame)

        os.system(f'cd "{e.out("samples")}" && convert -delay 20 -loop 0 *.jpg samples.gif')
        os.system(f'cd "{e.out("")}" && mv samples/samples.gif .')
