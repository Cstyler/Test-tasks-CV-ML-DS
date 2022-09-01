import itertools

import attr
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.image import Iterator


class CheckNumerics(Callback):
    def __init__(self, generator: Iterator):
        super(CheckNumerics, self).__init__()
        self.gen = generator

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and np.isnan(loss) or np.isinf(loss):
            print('On batch end. Batch %d: Invalid loss: %f' % (batch, loss))


@attr.s
class NormGradients(Callback):
    gen: Iterator = attr.ib()
    layer_name: str = attr.ib()
    batches_num: int = attr.ib()
    round_ndigits: int = attr.ib(default=4)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.gen.reset()
        grad_norm = self.calculate_target()
        self.gen.reset()
        logs[f'grad_norm_{self.layer_name}'] = round(grad_norm, self.round_ndigits)
        super().on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        model = self.model
        self.f = self.create_fun(model, self.layer_name)

    def create_fun(self, model, layer_name):
        nodes = [model.get_layer(layer_name).output]
        grads = model.optimizer.get_gradients(model.total_loss, nodes)
        summed_squares = [K.sum(K.square(g)) for g in grads]
        norm = K.sqrt(tf.cast(sum(summed_squares), tf.float32))
        symb_inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
        f = K.function(symb_inputs, [norm])
        return f

    def run_norm_fun(self, x, y):
        x_, y_, sample_weight_ = self.model._standardize_user_data(x, y)
        return self.f(x_ + y_ + sample_weight_)

    def calculate_target(self) -> float:
        metrics = []
        for batch_x, batch_y in itertools.islice(self.gen, self.batches_num):
            norm = self.run_norm_fun(batch_x, batch_y)
            metrics.append(norm)
        grad_norm = float(np.mean(metrics))
        return grad_norm
