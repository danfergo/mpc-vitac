import yaml

from ..experimenter import e

import time


def format_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds


class IncrementalMean:
    def __init__(self):
        self.total_sum = 0.0
        self.count = 0

    def update(self, value):
        self.total_sum += value
        self.count += 1
        return self.total_sum / self.count

    def get_mean(self):
        if self.count == 0:
            return None
        return self.total_sum / self.count


class Logger:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self, metrics_names=None):
        self.n_heads = len(e.loss) if isinstance(e.loss, tuple) else 1
        self.metrics_names = e['metrics_names']
        self.start_t = time.time()

        self.epoch_start_t = None
        self.incremental_epochs_mean_et = IncrementalMean()

    def on_epoch_start(self, ev):
        self.epoch_start_t = time.time()

    def on_validation_end(self, ev):
        history = ev['history']
        epoch = history['epoch']
        # Console log

        total_et = time.time() - self.start_t
        current_epoch_et = time.time() - self.epoch_start_t
        epochs_mean_et = self.incremental_epochs_mean_et.update(current_epoch_et)

        print('')
        print('')
        print('Done epoch ' + str(epoch + 1) + '.')

        hours, minutes, seconds = format_time(current_epoch_et)
        print("Current epoch met: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        hours, minutes, seconds = format_time(epochs_mean_et)
        print("Epochs mean et: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        hours, minutes, seconds = format_time(total_et)
        print("Total et: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        print('-----')
        # loss
        train_values = history['train']['losses_summary']
        val_values = history['val']['losses_summary']
        description = 'Loss'

        train_string = f'train {train_values[epoch][0]:.4f}'
        val_string = f'val {val_values[epoch][0]:.4f}' if 'val_loader' in e else ''

        print(f'\t{description.ljust(15)} \t\t{train_string.ljust(15)} \t{val_string.ljust(15)}')
        print('-----')

        for i in range(len(history['train']['head_losses_summary'])):
            train_values = history['train']['head_losses_summary'][i]
            val_values = history['val']['head_losses_summary'][i]
            description = f'Head loss: {i}'

            train_string = f'train {train_values[epoch][0]:.4f}'
            val_string = f'val {val_values[epoch][0]:.4f}' if 'val_loader' in e else ''

            print(f'\t{description.ljust(15)} \t\t{train_string.ljust(15)} \t{val_string.ljust(15)}')
        print('-----')

        # metrics
        for h in range(self.n_heads):
            print(f'Head {h}')
            for i in range(len(self.metrics_names)):
                ii = h * len(self.metrics_names) + i
                train_values = history['train']['metrics_summary'][ii]
                val_values = history['val']['metrics_summary'][ii]
                description = self.metrics_names[i]

                train_string = f'train {train_values[epoch][0]:.4f}'
                val_string = f'val {val_values[epoch][0]:.4f}' if 'val_loader' in e else ''

                print(f'\t{description.ljust(15)} \t\t{train_string.ljust(15)} \t{val_string.ljust(15)}')

        yaml.dump(ev['history'].data, open(e.out('stats.yaml'), 'w'))

        # File log
