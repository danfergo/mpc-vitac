import matplotlib.pyplot as plt
import torch

import numpy as np

from ..experimenter import e


class Plotter:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self, file_path='plot.png'):
        self.n_heads = len(e.loss) if isinstance(e.loss, tuple) else 1
        self.file_path = file_path
        self.metrics_names = e['metrics_names']

    def on_validation_end(self, ev):
        history = ev['history']

        plt.clf()
        plt.figure(figsize=(6, 20))

        idxs = list(range(ev['history']['epoch'] + 1))

        for i in range(1 + self.n_heads * len(self.metrics_names)):
            train_values = history['train']['losses_summary'] if i == 0 else history['train']['metrics_summary'][i - 1]
            val_values = history['val']['losses_summary'] if i == 0 else history['val']['metrics_summary'][i - 1]

            ni = i - 1 - ((i - 1) // len(self.metrics_names)) * len(self.metrics_names)
            description = 'Loss ' if i == 0 else self.metrics_names[ni]
            if i > 0:
                train_values = [x.cpu() if torch.is_tensor(x) else x for x in train_values]

            ax = plt.subplot(1 + self.n_heads * len(self.metrics_names), 1, i + 1)

            plt.plot([t[0] for t in train_values], label='Train ' + description if 'val_loader' in e else description)
            ax.fill_between(idxs, [t[2] for t in train_values], [t[3] for t in train_values], alpha=0.2)

            if 'val_loader' in e:
                if i > 0:
                    val_values = [x.cpu() if torch.is_tensor(x) else x for x in val_values]

                plt.plot([t[0] for t in val_values], label='Val ' + description)
                ax.fill_between(idxs, [t[0] - t[1] for t in val_values], [t[0] + t[1] for t in val_values], alpha=0.2)

            plt.ylabel(description)
            plt.legend()

        plt.savefig(e.out(self.file_path), dpi=150)
        plt.close()
