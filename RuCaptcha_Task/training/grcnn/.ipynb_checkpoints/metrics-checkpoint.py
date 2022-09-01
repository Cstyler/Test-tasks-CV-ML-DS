import operator
from collections import defaultdict
from pathlib import Path
from typing import Optional

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from pylibs import img_utils, pandas_utils, pyplot_utils, text_utils
from pylibs.numpy_utils import print_stats
from pylibs.pandas_utils import DF_FILE_FORMAT
from pylibs.types import FilterFunction
from .data_generation import InferenceBatchGenerator
from .predictor import PriceRecognizer
from .utils import PROJECT_NAME, calc_metric, load_model_grcnn, set_gpu, text_to_labels, decode_label

# ALPHABET = "".join(map(str, range(10))) + '-'
ALPHABET = "".join(map(str, range(10)))


def test_nn(dataset_dir_path: str, df_name: str, img_dir: str,
            model_num: int, epoch: int, batch_size: int, max_text_len: int,
            height: int, width: int, n_classes: int, grcl_fsize: int, grcl_niter: int, lstm_units: int,
            debug: bool = False,
            dist_filter: Optional[FilterFunction[float]] = None, text_filter: Optional[FilterFunction[str]] = None):
    gpu_id = 1080
    dataset_dir_path = Path(dataset_dir_path)
    images_dir_path = dataset_dir_path / img_dir

    test_set_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    test_df = pandas_utils.read_dataframe(test_set_path)
    test_df = test_df.sample(frac=1).head(30000)
    test_gen = InferenceBatchGenerator(test_df, batch_size,
                                       images_dir_path, width, height, max_text_len)

    set_gpu(gpu_id)

    model = load_model_grcnn(model_num, epoch, height, width,
                             n_classes, max_text_len, grcl_fsize,
                             grcl_niter, lstm_units)

    _, predictions = model.predict_generator(test_gen)
    # test_gen = BatchGenerator(test_df, 1, images_dir_path, max_text_len, morn_drop, shuffle=False)
    test_gen = InferenceBatchGenerator(test_df, 1,
                                       images_dir_path, width, height, max_text_len)

    dists = []
    dists_by_len = dict()
    rows = []
    for batch_x, prediction, (seg_id, row) in zip(test_gen, predictions, test_df.iterrows()):
        true_text = row['text']
        if not true_text:
            continue
        true_text = "".join(x for x in true_text if x in ALPHABET)
        text_len = len(true_text)
        if not text_filter or (text_filter and text_filter(true_text)):
            dist = calc_metric(prediction, true_text, ALPHABET, False, debug, dist_filter)
            if debug and dist_filter and dist_filter(dist):
                print("Seg id:", seg_id)
                show_image((batch_x['input'] + 1) / 2, (3, 10))
            if dist > 0:
                pred_text = decode_label(prediction.argmax(axis=1), ALPHABET)
                rows.append((seg_id, pred_text, true_text))
            true_num, len_count = dists_by_len.setdefault(text_len, (0, 0))
            dists_by_len[text_len] = true_num + (not dist), len_count + 1
            dists.append(dist)
    bad_samples_df = pd.DataFrame(rows, columns=('seg_id', 'pred', 'true')).set_index('seg_id')
    pandas_utils.write_dataframe(bad_samples_df, dataset_dir_path / (DF_FILE_FORMAT % 'bad_samples'))

    acc_by_len = accuracy_dict_from_count_dict(dists_by_len, 3)
    print(dists_by_len)
    print(acc_by_len)
    dists = np.asarray(dists)
    print_stats(dists)
    plt.hist(dists)
    plt.show()
    pyplot_utils.plot_hist_from_dict(acc_by_len)


def show_image(batch_x, figsize=None):
    for x in batch_x:
        img_utils.show_img(x, figsize)

def test_metric_by_markets(dataset_dir_path: str, df_name: str, img_dir: str,
                           model_num: int, epoch: int, batch_size: int, max_text_len: int,
                           height: int, width: int,
                           accuracy_threshold: float):
    model_dir = f'/srv/data_science/training/checkpoints/{PROJECT_NAME}/model{model_num}'
    dataset_dir_path = Path(dataset_dir_path)
    images_dir_path = dataset_dir_path / img_dir

    test_set_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    test_df = pandas_utils.read_dataframe(test_set_path)
    test_gen = InferenceBatchGenerator(test_df, batch_size,
                                       images_dir_path, width, height, max_text_len)
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    gpu_id = 1080
    set_gpu(gpu_id)
    rec = PriceRecognizer(model_num, epoch)
    pred_texts, pred_matrices = rec.recognize_batch(test_gen)
    dists = []
    true_labels = []
    true_lens = []
    dists_by_len = dict()
    dists_by_market = dict()
    dists_by_seg_type = dict()
    market_key = 'market_id'
    text_key = 'text'
    seg_type_key = 'seg_type'
    total = len(test_df.index)
    for pred_text, (tag_id, row) in tqdm(zip(pred_texts, test_df.iterrows()), total=total, smoothing=.01):
        true_text = row[text_key]
        if not true_text:
            continue
        true_text = "".join(x for x in true_text if x in ALPHABET)
        labels = text_to_labels(true_text, max_text_len, ALPHABET)
        true_labels.append(labels)
        true_lens.append(len(labels))
        text_len = len(true_text)
        dist = text_utils.levenshtein_distance_weighted(pred_text, true_text, 1, 2, 1)
        true_num, total_num = dists_by_len.setdefault(text_len, (0, 0))
        true_predict = not dist
        dists_by_len[text_len] = true_num + true_predict, total_num + 1
        seg_type = row[seg_type_key]
        true_num, total_num = dists_by_seg_type.setdefault(seg_type, (0, 0))
        dists_by_seg_type[seg_type] = true_num + true_predict, total_num + 1
        market = row[market_key]
        true_num, total_num = dists_by_market.setdefault(market, (0, 0))
        dists_by_market[market] = true_num + true_predict, total_num + 1
        dists.append(dist)
    ds_size = len(true_labels)
    costs = K.ctc_batch_cost(np.array(true_labels), pred_matrices,
                             np.ones((ds_size, 1)) * max_text_len, np.expand_dims(true_lens, 1))
    costs = K.squeeze(costs, 1)
    probs = K.get_value(K.exp(-costs))

    # probs_by_market = defaultdict(list)
    # for (_, row), dist, prob in zip(test_df.iterrows(), dists, probs):
    #     market = row[market_key]
    #     probs_by_market[market].append((dist, prob))
    #
    # thresholds = np.linspace(.0, np.max(probs), num=1000)
    # skipped_percents = dict()
    # for market, market_metrics in probs_by_market.items():
    #     thresholds_accuracy_pares = []
    #     ds_size = len(market_metrics)
    #     for thr in thresholds:
    #         true_num = 0
    #         total = 0
    #         for dist, prob in market_metrics:
    #             thr_flag = prob > thr
    #             total += thr_flag
    #             dist_flag = thr_flag and dist == 0
    #             true_num += dist_flag
    #         acc = true_num / total if total else .0
    #         skipped_percent = total / ds_size
    #         thresholds_accuracy_pares.append((round(acc * 100, 3), round(thr, 6), round(skipped_percent * 100, 3)))
    #     filtered_pairs = list(filter(lambda x: x[0] > 90. and x[2] > 1., thresholds_accuracy_pares))
    #     filtered_pairs = sorted(filtered_pairs, reverse=True, key=operator.itemgetter(2))
    #     best_triple = next((a, t, s) for a, t, s in filtered_pairs if a > accuracy_threshold)
    #     skipped_percents[market] = best_triple
    #
    # print(skipped_percents)

    probs_by_seg_type = defaultdict(lambda: defaultdict(list))
    for (_, row), dist, prob in zip(test_df.iterrows(), dists, probs):
        seg_type = row[seg_type_key]
        market = row[market_key]
        probs_by_seg_type[market][seg_type].append((dist, prob))

    thresholds = np.linspace(.0, np.max(probs), num=1000)
    for market, market_metrics in probs_by_seg_type.items():
        skipped_percents = dict()
        for seg_type, seg_metrics in market_metrics.items():
            thresholds_accuracy_pares = []
            ds_size = len(seg_metrics)
            for thr in thresholds:
                true_num = 0
                total = 0
                for dist, prob in seg_metrics:
                    thr_flag = prob > thr
                    total += thr_flag
                    dist_flag = thr_flag and dist == 0
                    true_num += dist_flag
                acc = true_num / total if total else .0
                skipped_percent = total / ds_size
                thresholds_accuracy_pares.append((round(acc * 100, 3), round(thr, 6), round(skipped_percent * 100, 3)))
            filtered_pairs = list(filter(lambda x: x[0] > 90. and x[2] > 1., thresholds_accuracy_pares))
            filtered_pairs = sorted(filtered_pairs, reverse=True, key=operator.itemgetter(2))
            best_triple = next((a, t, s) for a, t, s in filtered_pairs if a > accuracy_threshold)
            skipped_percents[seg_type] = best_triple
        print(market, skipped_percents)
    acc_by_len = accuracy_dict_from_count_dict(dists_by_len)
    acc_by_market = accuracy_dict_from_count_dict(dists_by_market)
    acc_by_seg_type = accuracy_dict_from_count_dict(dists_by_seg_type)
    print(dists_by_len)
    print(acc_by_len)
    print(dists_by_market)
    print(acc_by_market)
    print(dists_by_seg_type)
    print(acc_by_seg_type)
    dists = np.asarray(dists)
    print_stats(dists)
    print_stats(probs)
    plt.hist(dists)
    plt.show()
    plt.hist(np.round(probs, 7), bins=(.0001, .001, .002, .003, .004, .005, .01, .015, .02, .03))
    plt.show()
    pyplot_utils.plot_hist_from_dict(acc_by_len)


def accuracy_dict_from_count_dict(count_dict: dict, round_ndigits=3) -> dict:
    acc_dict = dict()
    for k, (true_num, total_num) in count_dict.items():
        if total_num:
            acc_dict[k] = round(true_num / total_num, round_ndigits)
    return acc_dict


def test_google(dataset_dir_path: str, df_name: str):
    dataset_dir_path = Path(dataset_dir_path)
    dataset_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(dataset_path)
    dists = []
    dists_by_len = dict()
    default_tuple = (0, 0)

    for _, row in df.iterrows():
        true_text = row['text']
        if not true_text:
            continue
        pred_text = row['google_text']
        true_text = "".join(x for x in true_text if x in ALPHABET)
        text_len = len(true_text)
        dist = text_utils.levenshtein_distance_weighted(pred_text, true_text, 1, 2, 1)
        true_num, len_count = dists_by_len.setdefault(text_len, default_tuple)
        dists_by_len[text_len] = true_num + (not dist), len_count + 1
        dists.append(dist)

    acc_by_len = dict()
    for len_, (true_num, count) in dists_by_len.items():
        acc_by_len[len_] = round(true_num / count, 3)
    print(dists_by_len)
    print(acc_by_len)
    dists = np.asarray(dists)
    print_stats(dists)
    plt.hist(dists)
    plt.show()
    pyplot_utils.plot_hist_from_dict(acc_by_len)
