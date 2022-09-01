import functools
from typing import Dict

import keras
import numpy as np
from keras.models import Model
from tqdm import tqdm_notebook

from pylibs.img_utils import show_img
from pylibs.keras import keract
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from .callbacks import MetricCallback
from .data_generation import BatchGenerator
from .predictor import ALPHABET
from .utils import calc_metric


def data_generator(image_dir, batch_size, max_text_len, df_name, morn_drop):
    dataset_dir_str = "/srv/data_science/storage/product_code_ocr"
    dataset_dir_path = Path(dataset_dir_str)
    images_dir_path = dataset_dir_path / image_dir
    train_set_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    train_df = read_dataframe(train_set_path)
    # train_gen = BatchGenerator(train_df, batch_size, images_dir_path, max_text_len,
    #                            transform_list, train_augment_prob, seed=seed)
    train_gen = BatchGenerator(train_df, batch_size, images_dir_path, max_text_len, morn_drop)
    return train_gen


def train_batch(model, batch_x, batch_y, debug=False):
    n_iter = 6000
    for _ in tqdm_notebook(range(n_iter)):
        leven = predict_batch(model, batch_x, debug)
        loss = model.train_on_batch(batch_x, batch_y)
        print(f"%.3f %.3f" % (leven, loss[0]), sep=' ', end='\r', flush=True)


def predict_batch(model: Model, batch_x: Dict[str, np.ndarray], debug=False) -> float:
    loss, pred = model.predict_on_batch(batch_x)
    acts = get_model_activations(model, batch_x, 'morn')
    dists = []
    figsize = (3, 10)
    for sample, text, label, img, act in zip(pred, batch_x['texts'],
                                             batch_x['labels'], batch_x['input'], acts):
        dist = calc_metric(sample, text, ALPHABET, debug=debug)
        dists.append(dist)
        if debug and dist > 0:
            show_img(preprocess(img), figsize)
            show_img(preprocess(act), figsize)
    mean_score = np.mean(dists)
    return float(mean_score)


def preprocess(x: np.ndarray) -> np.ndarray:
    return (x + 1) / 2


def get_model_activations(model: Model, batch_x: Dict[str, np.ndarray], layer_name: str):
    act = keract.get_activations(model, batch_x, layer_name)
    x_rectified = list(act.items())[0][1]
    return x_rectified


def freeze_recognizer(model: Model):
    for layer in model.layers:
        if not layer.name == 'morn' and layer.trainable:
            layer.trainable = False
    compile_model(model)


def compile_model(model: Model):
    lr = 0.5
    optimizer = keras.optimizers.Adadelta(lr=lr, rho=.9)

    def loss_fun(y_true, y_pred):
        return y_pred

    # loss_dict = dict(ctc=loss_fun, predictions=None)
    loss_dict = dict(total_loss=loss_fun, predictions=None)
    model.compile(loss=loss_dict, optimizer=optimizer)


def predict_str(label):
    return "".join('•' if x == len(ALPHABET) else str(x) for x in map(int, label))


def run_callback(gen: BatchGenerator, model: Model):
    metric_function = functools.partial(calc_metric, alphabet=ALPHABET)
    callback = MetricCallback(gen, metric_function, 'train', 'texts', 10)
    callback.set_model(model)
    logs = dict(loss=0)
    epoch = 0
    callback.on_epoch_end(epoch, logs)
    print(logs)
