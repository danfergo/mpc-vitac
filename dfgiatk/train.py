"""
Main class that is used to train the NN i.e. run the optimization loop. Samples batches from the train data,
feeds to the NN, computes gradients and updates the network weights. Then computes training metrics. Per epoch,
after training, runs the validation loop by sampling batches from the validation data, and computes validation
metrics. At each relevant moment, calls the corresponding method of the History object, then calls the callbacks
passing them the history.
"""

from .experimenter import e
import torch

torch.cuda.empty_cache()


def feed_chunk(chunk, loss_factor, compute_loss=True, update_weights=True, zero_grad=True, step=True,
               return_head_losses=False):
    model = e.model
    optimizer = e.optimizer if update_weights else None
    loss_fn = e.loss if compute_loss else None
    complex_loss = e.complex_loss if 'complex_loss' in e else False

    if compute_loss and update_weights:
        # zero the parameter gradients
        if zero_grad:
            optimizer.zero_grad()

    # get predictions and computes loss
    x, y_true = chunk
    if isinstance(x, tuple):
        y_pred = model(*x)  # expand inputs in case of multiple
    else:
        y_pred = model(x)

    if compute_loss:
        if isinstance(y_pred, tuple):
            losses = [
                loss_factor * (loss_fn[i](x, y_true, y_pred) if complex_loss else loss_fn[i](y_pred[i], y_true[i]))
                for i in range(len(y_true))
            ]
            loss = torch.stack(losses).mean()
        else:
            loss = loss_factor * (loss_fn(x, y_true, y_pred) if complex_loss else loss_fn(y_pred, y_true))
            losses = [loss]

        if update_weights:

            # compute gradients and
            loss.backward()

            # perform optimization step
            if step:
                optimizer.step()
        if return_head_losses:
            return y_pred, loss, losses
        else:
            return y_pred, loss
    return y_pred


def feed_batch(batch, compute_loss=True, feed_size=None, update_weights=True, return_head_losses=False):
    batch_size = batch[0][0].size()[0] if isinstance(batch[0], tuple) else batch[0].size()[0]
    feed_size = feed_size or e.feed_size or batch_size
    batch_loss = 0
    batch_pred = []
    head_losses = []

    for c in range(0, batch_size, feed_size):
        c0 = c
        c1 = min(c + feed_size, batch_size)
        loss_factor = (c1 - c0) / batch_size
        ret = feed_chunk(
            tuple([
                tuple([tt[c0:c1, ...] for tt in t])  # slices a chunk of each xs or ys
                if isinstance(t, tuple)  # in case of multiple/multi-modal inputs/xs outputs/ys
                else t[c0:c1, ...]  # slices a chunk of the x or y
                for t in list(batch)  # batch is a tuple of xs, ys
            ]),
            compute_loss=compute_loss,
            loss_factor=loss_factor,
            update_weights=update_weights,
            zero_grad=c == 0,  # first step
            step=c1 == batch_size,  # last step,
            return_head_losses=return_head_losses
        )
        if compute_loss:
            batch_loss += ret[1].item()

            if return_head_losses:
                head_losses = (head_losses or []) + [ret[2]]

        c_pred = ret[0] if compute_loss else ret
        batch_pred = (batch_pred or []) + [c_pred]

    batch_pred = tuple([torch.cat(batch_pred_head) for batch_pred_head in zip(*batch_pred)]) \
        if isinstance(batch[1], tuple) \
        else torch.cat(tuple(batch_pred))

    if compute_loss:
        if return_head_losses:
            head_losses = tuple([sum(hl).item() for hl in zip(*head_losses)])
            return batch_loss, batch_pred, head_losses
        else:
            return batch_loss, batch_pred

    return batch_pred


def fit_to_batch(batch, return_head_losses=True):
    return feed_batch(batch, update_weights=True, return_head_losses=return_head_losses)


def predict_batch(batch, compute_loss=True, feed_size=None, return_head_losses=False):
    return feed_batch(batch, compute_loss, feed_size, update_weights=False, return_head_losses=return_head_losses)


def fit_to_dataset():
    """
    The optimization loop per se
    :return:
    """

    epochs, batch_size, batches_per_epoch, data_loader, train_device, model = e[
        'epochs',
        'batch_size',
        'batches_per_epoch',
        'data_loader',
        'train_device',
        'model'
    ]

    model.to(train_device)

    for epoch in range(epochs):
        e.emit('epoch_start', {'epoch': epoch})

        # Train some batches
        for batch in iter(data_loader):
            e.emit('train_batch_start')

            # batch = next(x)
            batch_loss, batch_pred, head_losses = fit_to_batch(batch)

            # save running stats
            e.emit('train_batch_end', {'batch': batch,
                                       'y_pred': batch_pred,
                                       'loss': batch_loss,
                                       'head_losses': head_losses
                                       })

        e.emit('epoch_end', {'n_used_batches': batches_per_epoch})
