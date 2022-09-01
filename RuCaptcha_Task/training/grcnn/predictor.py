from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from keras.layers import (BatchNormalization, Bidirectional, Conv2D, Input, LSTM, MaxPool2D, ReLU, Reshape,
                          Softmax, ZeroPadding2D)
from keras.models import Model

from .model import GRCL
from .utils import decode_label, find_model_path

ALPHABET = "".join(map(str, range(10)))
width, height = 64, 32
n_classes = 10
max_text_len = 17
BATCH_SIZE = 8


def normalize_img(img: np.ndarray) -> np.ndarray:
    img = 2 * (img / 255) - 1
    return img.astype(np.float32)


def process_img(img: np.ndarray) -> np.ndarray:
    img = normalize_img(img)
    img = cv2.resize(img, (width, height))
    return img


relu = ReLU()


def GRCNN(grcl_niter, grcl_fsize, lstm_units) -> Model:
    pool_size = 2
    input_layer = Input(name="input", shape=(height, width, 3))
    x = Conv2D(64, 3, padding='same')(input_layer)
    x = BatchNormalization()(x)  # TODO it was not here
    x = relu(x)
    x = MaxPool2D(pool_size=pool_size, strides=2)(x)
    x = GRCL(x, 64, grcl_niter, grcl_fsize)
    x = MaxPool2D(pool_size=pool_size, strides=2)(x)
    x = GRCL(x, 128, grcl_niter, grcl_fsize)
    x = ZeroPadding2D(padding=(0, 1))(x)
    strides = (2, 1)
    x = MaxPool2D(pool_size=pool_size, strides=strides)(x)
    x = GRCL(x, 256, grcl_niter, grcl_fsize)
    x = ZeroPadding2D(padding=(0, 1))(x)
    strides = (2, 1)  # in orig (2, 1)
    x = MaxPool2D(pool_size=pool_size, strides=strides)(x)
    x = Conv2D(512, 2)(x)  # no padding
    x = BatchNormalization()(x)
    x = relu(x)

    w = int(x.shape[1] * x.shape[2])  # in orig x.shape[1] == 1
    h = int(x.shape[3])
    x = Reshape((w, h))(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum')(x)
    units = n_classes + 1
    x = Bidirectional(LSTM(units, return_sequences=True), merge_mode='sum')(x)
    y_pred = Softmax(name='predictions')(x)
    model = Model(inputs=input_layer, outputs=y_pred)
    return model


class PriceRecognizer:
    def __init__(self, model_num: int, epoch: int):
        model_path = find_model_path(model_num, epoch)
        self.model = GRCNN(5, 3, 512)
        self.model.load_weights(model_path)

    def recognize_batch(self, images: Iterable[np.ndarray]) -> Iterable[str]:
        x = np.stack(tuple(map(process_img, images)))
        predictions = self.model.predict(x, BATCH_SIZE)
        return map(self.get_text_from_nn_output, predictions)

    def recognize_batch(self, images: Iterable[dict]) -> Tuple[Iterable[str], np.ndarray]:
        predictions = self.model.predict_generator(images)
        return map(self.get_text_from_nn_output, predictions), predictions

    def recognize(self, image: np.ndarray) -> str:
        x = np.expand_dims(normalize_img(image), -1)
        pred = self.model.predict(x, 1)
        return self.get_text_from_nn_output(pred)

    @staticmethod
    def get_text_from_nn_output(pred: np.ndarray) -> str:
        pred_argmax = pred.argmax(axis=1)
        pred_text = decode_label(pred_argmax, ALPHABET)
        return pred_text
