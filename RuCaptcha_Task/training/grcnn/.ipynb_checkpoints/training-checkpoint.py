import random

import albumentations as albu
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Optimizer

from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from .callbacks import get_callbacks
from .data_generation import BatchGenerator
from .utils import DATASET_DIR, find_model_path, set_gpu

seed = 42
gpu_id = 1080
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)


def loss_fun(y_true, y_pred):
    return y_pred


def compile_model(model: Model, optimizer: Optimizer, loss_key: str):
    loss_dict = dict(predictions=None)
    loss_dict[loss_key] = loss_fun
    model.compile(loss=loss_dict, optimizer=optimizer)


def get_transforms():
    WHITE_PIXEL = 255
    ps = [.5, .5]
    # ps = [.75, .75]
    transform_list = [
            albu.OneOf(
                    [
                            # pixel augmentations
                            albu.IAAAdditiveGaussianNoise(loc=0,
                                                          scale=(.01 * WHITE_PIXEL, .05 * WHITE_PIXEL), p=1),
                            albu.RandomBrightnessContrast(contrast_limit=.2,
                                                          brightness_limit=.2, p=1),
                            albu.RandomGamma(gamma_limit=(80, 120), p=1),
                    ],
                    p=ps[0],
            ),
            albu.OneOf(
                    [
                            # convolutional augmentations
                            albu.IAASharpen(alpha=(.05, .2), lightness=(.91, 1.), p=1),
                            albu.Blur(blur_limit=3, p=1),
                            albu.MotionBlur(blur_limit=3, p=1),
                            albu.MedianBlur(blur_limit=3, p=1),
                    ],
                    p=ps[1],
            )
    ]
    return transform_list


def train(train_df_name, val_df_name, img_dir, model_num, initial_epoch, epochs, train_augment_prob, batch_size,
          pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units, loss_weights=None):
    transform_list = get_transforms()
    images_dir_path = DATASET_DIR / img_dir

    train_set_path = DATASET_DIR / (DF_FILE_FORMAT % train_df_name)
    train_df = read_dataframe(train_set_path)
    train_gen = BatchGenerator(train_df, batch_size, images_dir_path, max_text_len,
                               transform_list, train_augment_prob, seed=seed)

    val_set_path = DATASET_DIR / (DF_FILE_FORMAT % val_df_name)
    val_df = read_dataframe(val_set_path)
    val_gen = BatchGenerator(val_df, batch_size, images_dir_path, max_text_len, shuffle=False)

    set_gpu(gpu_id)

    callbacks = get_callbacks(model_num, train_gen, val_gen)

    # TODO change this to model config : dict str->value(float, sequence[float], etc.)
    if loss_weights:
        from .model import GRCNN_mse
        loss_key = 'total_loss'
        model = GRCNN_mse(height, width, n_classes, max_text_len, grcl_fsize, grcl_niter, lstm_units, loss_weights)
    else:
        from .model import GRCNN
        loss_key = 'ctc'
        model = GRCNN(height, width, n_classes, max_text_len, grcl_fsize, grcl_niter, lstm_units)

    if initial_epoch:
        model_file_path = find_model_path(model_num, initial_epoch)
        print(f"Model found: {model_file_path}")
        model.load_weights(model_file_path)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)
    compile_model(model, optimizer, loss_key)

    model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch, epochs=epochs,
                        verbose=1, callbacks=callbacks, use_multiprocessing=True,
                        workers=3, max_queue_size=20, initial_epoch=initial_epoch,
                        validation_data=val_gen, validation_steps=val_gen.steps_per_epoch)
