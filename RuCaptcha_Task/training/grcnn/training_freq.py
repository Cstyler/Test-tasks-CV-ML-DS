import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from pylibs.numpy_utils import read_array
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from .callbacks import get_callbacks
from .data_generation import BatchGeneratorFreqModel
from .model import conv_freq_model
from .utils import DATASET_DIR, find_model_path

seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)


def compile_model(model: Model, optimizer: Optimizer):
    loss_dict = {"freq": mean_squared_error}
    model.compile(loss=loss_dict, optimizer=optimizer)


def get_transforms():
    WHITE_PIXEL = 255
    ps = [.5, .5]
    # ps = [.75, .75]
    transform_list = [
        # albu.OneOf(
        #         [
        #                 # pixel augmentations
        #                 albu.IAAAdditiveGaussianNoise(loc=0,
        #                                               scale=(.01 * WHITE_PIXEL, .05 * WHITE_PIXEL), p=1),
        #                 albu.RandomBrightnessContrast(contrast_limit=.2,
        #                                               brightness_limit=.2, p=1),
        #                 albu.RandomGamma(gamma_limit=(80, 120), p=1),
        #         ],
        #         p=ps[0],
        # ),
        # albu.OneOf(
        #         [
        #                 # convolutional augmentations
        #                 albu.IAASharpen(alpha=(.05, .2), lightness=(.91, 1.), p=1),
        #                 albu.Blur(blur_limit=3, p=1),
        #                 albu.MotionBlur(blur_limit=3, p=1),
        #                 albu.MedianBlur(blur_limit=3, p=1),
        #         ],
        #         p=ps[1],
        # )
    ]
    return transform_list


def train(train_df_name, val_df_name, img_file, model_num, initial_epoch, epochs, train_augment_prob, batch_size,
          pretrain_model_path, optimizer, width, height, max_text_len, n_classes):
    transform_list = get_transforms()
    imgs = read_array(DATASET_DIR.parent / img_file)

    train_set_path = DATASET_DIR / (DF_FILE_FORMAT % train_df_name)
    train_df = read_dataframe(train_set_path)
    train_gen = BatchGeneratorFreqModel(train_df, batch_size, imgs, max_text_len,
                                        transform_list, train_augment_prob, seed=seed)

    val_set_path = DATASET_DIR / (DF_FILE_FORMAT % val_df_name)
    val_df = read_dataframe(val_set_path)
    val_gen = BatchGeneratorFreqModel(val_df, batch_size, imgs, max_text_len, shuffle=False)

    callbacks = get_callbacks(model_num, train_gen, val_gen)

    # TODO change this to model config : dict str->value(float, sequence[float], etc.)=
    model = conv_freq_model(height, width, n_classes, max_text_len)

    if initial_epoch:
        model_file_path = find_model_path(model_num, initial_epoch)
        print(f"Model found: {model_file_path}")
        model.load_weights(model_file_path)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)
    compile_model(model, optimizer)

    model.fit(train_gen, steps_per_epoch=train_gen.steps_per_epoch, epochs=epochs,
              verbose=1, callbacks=callbacks, initial_epoch=initial_epoch,
              validation_data=val_gen, validation_steps=val_gen.steps_per_epoch)
