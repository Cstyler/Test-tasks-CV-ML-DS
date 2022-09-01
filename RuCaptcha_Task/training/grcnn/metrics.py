from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pylibs import img_utils, pandas_utils
from pylibs.numpy_utils import print_stats, read_array
from pylibs.pandas_utils import DF_FILE_FORMAT
from pylibs.types import FilterFunction
from .data_generation import InferenceBatchGenerator
from .utils import calc_metric, load_model_grcnn, decode_label, DATASET_DIR

ALPHABET = "".join(map(str, range(10)))


def test_nn(df_name: str, img_file: str,
            model_num: int, epoch: int, batch_size: int, max_text_len: int,
            height: int, width: int, n_classes: int, grcl_fsize: int, grcl_niter: int, lstm_units: int,
            debug: bool = False,
            dist_filter: Optional[FilterFunction[float]] = None, text_filter: Optional[FilterFunction[str]] = None):
    images = read_array(DATASET_DIR.parent / img_file)

    test_set_path = DATASET_DIR / (DF_FILE_FORMAT % df_name)
    test_df = pandas_utils.read_dataframe(test_set_path)
    # test_df = test_df.sample(frac=1).head(20)
    test_gen = InferenceBatchGenerator(test_df, batch_size,
                                       images, width, height, max_text_len)

    model = load_model_grcnn(model_num, epoch, height, width,
                             n_classes, max_text_len, grcl_fsize,
                             grcl_niter, lstm_units)

    _, predictions = model.predict_generator(test_gen)
    test_gen = InferenceBatchGenerator(test_df, 1,
                                       images, width, height, max_text_len)

    dists = []
    rows = []
    for batch_x, prediction, (img_id, row) in zip(test_gen, predictions, test_df.iterrows()):
        true_text = row['text']
        if not true_text:
            continue
        true_text = "".join(x for x in true_text if x in ALPHABET)
        if not text_filter or (text_filter and text_filter(true_text)):
            dist = calc_metric(prediction, true_text, ALPHABET, False, debug, dist_filter)
            if debug and dist_filter and dist_filter(dist):
                print("Img id:", img_id)
                img_ = (((batch_x['input'] + 1) / 2) * 255).astype(np.uint8)
                show_image(img_, (3, 10))
            if dist > 0:
                pred_text = decode_label(prediction.argmax(axis=1), ALPHABET)
                rows.append((img_id, pred_text, true_text))
            dists.append(dist)
    bad_samples_df = pd.DataFrame(rows, columns=('img_id', 'pred', 'true')).set_index('img_id')
    pandas_utils.write_dataframe(bad_samples_df, DATASET_DIR / (DF_FILE_FORMAT % 'bad_samples'))

    dists = np.asarray(dists)
    print(np.sum(dists == 0) / len(dists))
    print_stats(dists)
    plt.hist(dists)
    plt.show()


def show_image(batch_x, figsize=None):
    for x in batch_x:
        img_utils.show_img(x, figsize)


def accuracy_dict_from_count_dict(count_dict: dict, round_ndigits=3) -> dict:
    acc_dict = dict()
    for k, (true_num, total_num) in count_dict.items():
        if total_num:
            acc_dict[k] = round(true_num / total_num, round_ndigits)
    return acc_dict
