import operator
import os
from pathlib import Path

import numpy as np
from pylibs.storage_utils import get_file_sharding
from tqdm import tqdm_notebook as tqdm

from pylibs.jpeg_utils import get_shape
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe


def check_df_imgs_exist(ds_dir: Path, df_name: str, img_dir: str):
    df_path = ds_dir / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)
    images_dir = ds_dir / img_dir
    count = 0
    for tag_id, row in tqdm(df.iterrows(), total=len(df.index), smoothing=.01):
        p = get_file_sharding(images_dir, tag_id)
        if not p.exists():
            count += 1
    print("Doesn't exist count:", count)


def shape_analysis(ds_dir: Path, df_name: str, img_dir: str, max_height: int = 32):
    df_path = ds_dir / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)
    images_dir = ds_dir / img_dir

    widths = []
    text_lens = []
    for tag_id, row in tqdm(df.iterrows(), total=len(df.index), smoothing=.01):
        img_path = get_file_sharding(images_dir, tag_id)
        w, h = get_shape(img_path)
        target_w = int(max_height / h * w)
        widths.append(target_w)
        text_len = len(row['text'])
        text_lens.append(text_len)
    return np.array(widths), np.array(text_lens)


def check_images(ds_dir: Path, df_name: str, img_dir: str):
    df_path = ds_dir / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)
    images_dir = ds_dir / img_dir
    recorded_tags = set(df.index)
    c = 0
    for root, dirs, files in tqdm(os.walk(images_dir), total=16 ** 3, smoothing=.01):
        seg_ids = set(map(operator.attrgetter('stem'), map(Path, files)))
        c += len(seg_ids) - len(seg_ids & recorded_tags)
    print("%s images are not recorded in dfs" % c)
