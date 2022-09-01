import random

import albumentations as albu
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib

from . import model_lib

device_lib.list_local_devices()
import cv2
from .callbacks import get_callbacks
from .data_generation import BatchGenerator
from .utils import DATASET_DIR, ARR_FILE_FORMAT

seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)


def get_transforms(height, width):
    WHITE_PIXEL = 255
    ps = [.5, .5, .5, .2, .2]
    transform_list = [
        albu.HorizontalFlip(p=ps[0]),
        albu.ShiftScaleRotate(shift_limit=.05, scale_limit=.05,
                              rotate_limit=5, border_mode=cv2.BORDER_REFLECT_101,
                              interpolation=cv2.INTER_LINEAR, p=ps[1]),
        albu.RandomResizedCrop(height, width, scale=(.7, 1), ratio=(1.25, 1.78), p=ps[2]),
        albu.OneOf(
            [
                # pixel augmentations
                albu.IAAAdditiveGaussianNoise(loc=0,
                                              scale=(.01 * WHITE_PIXEL, .05 * WHITE_PIXEL), p=1),
                albu.RandomBrightnessContrast(contrast_limit=.2,
                                              brightness_limit=.2, p=1),
                albu.RandomGamma(gamma_limit=(80, 120), p=1),
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
                albu.HueSaturationValue(p=1)
            ],
            p=ps[3],
        ),
        albu.OneOf(
            [
                # convolutional augmentations
                albu.IAASharpen(alpha=(.05, .2), lightness=(.91, 1.), p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
                albu.MedianBlur(blur_limit=3, p=1),
            ],
            p=ps[4],
        )
    ]
    return transform_list


def compile_model(model: Model, optimizer, class_weights: list):
    # classes: background, class1, class2
    dice_loss = model_lib.losses.DiceLoss(class_weights=np.array(class_weights))

    focal_loss = model_lib.losses.CategoricalFocalLoss()
    total_loss = dice_loss + focal_loss
    # ce_loss = model_lib.losses.CategoricalCELoss()
    # total_loss = dice_loss + ce_loss
    thr = 0.5
    class_indexes = [1, 2, 3]
    metrics = [model_lib.metrics.IOUScore(threshold=thr, class_indexes=class_indexes),
               model_lib.metrics.FScore(threshold=thr, class_indexes=class_indexes),
               model_lib.metrics.Precision(threshold=thr, class_indexes=class_indexes),
               model_lib.metrics.Recall(threshold=thr, class_indexes=class_indexes)]

    model.compile(optimizer, total_loss, metrics)


def train(train_img_arr_name, val_img_arr_name, model_num, initial_epoch, epochs, train_augment_prob, batch_size,
          pretrain_model_path, optimizer, width, height, n_classes, base_lr, max_lr, class_weights):
    transform_list = get_transforms(height, width)

    img_arr_path = DATASET_DIR / (ARR_FILE_FORMAT % train_img_arr_name)
    mask_arr_path = DATASET_DIR / (ARR_FILE_FORMAT % (train_img_arr_name + '_masks'))
    train_gen = BatchGenerator(img_arr_path, mask_arr_path, batch_size, transform_list, train_augment_prob, seed=seed)

    img_arr_path = DATASET_DIR / (ARR_FILE_FORMAT % val_img_arr_name)
    mask_arr_path = DATASET_DIR / (ARR_FILE_FORMAT % (val_img_arr_name + '_masks'))
    val_gen = BatchGenerator(img_arr_path, mask_arr_path, batch_size, shuffle=False)

    cycle_epochs = 8
    cyclic_lr_step_size = train_gen.steps_per_epoch * cycle_epochs
    callbacks = get_callbacks(model_num, base_lr, max_lr, cyclic_lr_step_size)

    # create model_lib
    backbone = 'resnext101'
    if pretrain_model_path:
        model: Model = model_lib.Unet(backbone, classes=n_classes, activation='softmax',
                               weights=pretrain_model_path, encoder_weights=None)
    else:
        model: Model = model_lib.Unet(backbone, classes=n_classes,
                               activation='softmax', encoder_weights='imagenet')

    compile_model(model, optimizer, class_weights)

    # if initial_epoch:
    #     model_file_path = find_model_path(model_num, initial_epoch)
    #     print(f"Model found: {model_file_path}")
    #     model_lib.load_weights(model_file_path)

    model.fit(train_gen, steps_per_epoch=train_gen.steps_per_epoch, epochs=epochs,
              verbose=1, callbacks=callbacks, initial_epoch=initial_epoch,
              validation_data=val_gen, validation_steps=val_gen.steps_per_epoch)
