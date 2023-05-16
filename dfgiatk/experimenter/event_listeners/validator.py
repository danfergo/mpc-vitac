from ..experimenter import e
from ...ops.img import cvt_batch, CVT_CHW2HWC
from ...train import predict_batch

import torch
import numpy as np

import cv2


class Stats:
    def __init__(self, n_metrics, n_heads=1):
        self.n_metrics = n_metrics
        self.n_heads = n_heads

        self.data = {
            'epoch': 0,
            'train': {
                'losses': 0.0,
                'metrics': [],
                'head_losses': [],
                'losses_summary': [],
                'metrics_summary': [[] for _ in range(n_metrics * n_heads)],
                'head_losses_summary': [[] for _ in range(n_heads)],
                'all_losses': [],
                'all_metrics': [[] for _ in range(n_metrics * n_heads)],
                'all_head_losses': [[] for _ in range(n_heads)],
            },
            'val': {
                'loss': 0.0,
                'metrics': [],
                'losses_summary': [],
                'metrics_summary': [[] for _ in range(n_metrics * n_heads)],
                'head_losses_summary': [[] for _ in range(n_heads)],
                'all_losses': [],
                'all_metrics': [[] for _ in range(n_metrics * n_heads)],
                'all_head_losses': [[] for _ in range(n_heads)]
            }
        }

    def set_current_epoch(self, epoch):
        self['epoch'] = epoch

    def reset_running_stats(self, phase):
        self[phase]['losses'] = []
        self[phase]['metrics'] = [[] for _ in range(self.n_metrics * self.n_heads)]
        self[phase]['head_losses'] = [[] for _ in range(self.n_heads)]

    def update_running_stats(self, phase, loss, metrics, head_losses):
        self[phase]['losses'].append(loss)
        [self[phase]['metrics'][m].append(metrics[m]) for m in range(len(metrics))]
        [self[phase]['head_losses'][m].append(head_losses[m]) for m in range(len(head_losses))]

    def save_running_stats(self, phase):
        def stats(x):
            return [np.mean(x), np.std(x), np.min(x), np.max(x)]

        self[phase]['all_losses'].append(self[phase]['losses'])
        self[phase]['losses_summary'].append(stats(self[phase]['losses']))

        [
            self[phase]['all_metrics'][i]
                .append(self[phase]['metrics'][i])
            for i in range(self.n_metrics * self.n_heads)
        ]
        [
            self[phase]['metrics_summary'][i]
                .append(stats(self[phase]['metrics'][i]))
            for i in range(self.n_metrics * self.n_heads)
        ]

        [
            self[phase]['all_head_losses'][i]
                .append(self[phase]['head_losses'][i])
            for i in range(self.n_heads)
        ]

        [
            self[phase]['head_losses_summary'][i]
                .append(stats(self[phase]['head_losses'][i]))
            for i in range(self.n_heads)
        ]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]


class Validator:

    def __init__(self):
        self.n_heads = len(e.loss) if isinstance(e.loss, tuple) else 1
        self.stats = Stats(len(e.metrics), self.n_heads)
        self.metrics = e.metrics
        self.loss_fn = e.loss
        self.loader = e.val_loader if 'val_loader' in e else None
        self.n_val_batches = e.n_val_batches if 'val_loader' in e else None

    def on_epoch_start(self, ev):
        self.stats.set_current_epoch(ev['epoch'])
        self.stats.reset_running_stats('train')

    def on_train_batch_end(self, ev):
        x, y_true = ev['batch']
        y_pred = ev['y_pred']

        if self.n_heads > 1:
            metrics_results = [m(y_pred[i], y_true[i]).item() for i in range(self.n_heads) for m in self.metrics]
        else:
            metrics_results = [m(y_pred, y_true).item() for m in self.metrics]

        self.stats.update_running_stats('train', ev['loss'], metrics_results, ev['head_losses'])

    def on_epoch_end(self, ev):
        self.stats.save_running_stats('train')
        self.stats.reset_running_stats('val')

        if self.loader is not None:
            with torch.inference_mode():

                for batch in iter(self.loader):
                    x, y_true = batch
                    loss, y_pred, head_losses = predict_batch(batch, feed_size=e.val_feed_size, return_head_losses=True)

                    self.stats.update_running_stats('val',
                                                    loss,
                                                    [m(y_pred[i], y_true[i]).item() for i in range(2) for m in
                                                     self.metrics],
                                                    head_losses
                                                    )

                    # xx = cvt_batch(x[0].cpu().numpy(), CVT_CHW2HWC)
                    # yy_true = y_true.cpu().numpy()
                    # yy_pred = y_pred.cpu().numpy()
                    #
                    # for i in range(len(xx)):
                    #     cv2.imshow('r', xx[i])
                    #     print(yy_true[i], np.argmax(yy_pred[i]))
                    #     cv2.waitKey(-1)
                    #
                    # print(y_pred)

            self.stats.save_running_stats('val')

        e.emit('validation_end', {'history': self.stats})
