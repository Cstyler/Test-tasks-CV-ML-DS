from pathlib import Path

import cv2
import numpy as np
from pylibs.storage_utils import get_file_sharding, save_file_sharding
from tqdm import tqdm_notebook as tqdm

from pylibs import img_utils, jpeg_utils, pandas_utils
from pylibs.pandas_utils import DF_FILE_FORMAT

int_dtype = 'uint8'


def resize_images(dataset_dir_path: str, df_name: str, img_dir: str, save_img_dir: str,
                  width: int, height: int, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    source_img_dir = dataset_dir_path / img_dir
    img_resized_dir_path = dataset_dir_path / save_img_dir
    img_resized_dir_path.mkdir(exist_ok=True)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)

    shape = (width, height)
    index = df.index
    for tag_id in tqdm(index, total=len(index), smoothing=.01):
        img_path = get_file_sharding(source_img_dir, tag_id)
        save_img_path = save_file_sharding(img_resized_dir_path, img_path)
        if save_img_path.exists():
            continue
        try:
            img = jpeg_utils.read_jpeg(img_path)
        except (OSError, IOError):
            continue
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        if debug:
            img_utils.show_img(img, (3, 5))
        else:
            jpeg_utils.write_jpeg(save_img_path, img)


def resize_images_padding(dataset_dir_path: str, df_name: str, img_dir: str, save_img_dir: str,
                          width: int, height: int, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    source_img_dir = dataset_dir_path / img_dir
    img_resized_dir_path = dataset_dir_path / save_img_dir
    img_resized_dir_path.mkdir(exist_ok=True)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)

    index = df.index
    for tag_id in tqdm(index, total=len(index), smoothing=.01):
        img_path = get_file_sharding(source_img_dir, tag_id)
        save_img_path = save_file_sharding(img_resized_dir_path, img_path)
        if save_img_path.exists():
            continue
        try:
            img = jpeg_utils.read_jpeg(img_path)
        except (OSError, IOError):
            continue
        h, w, _ = img.shape
        target_w = int(height / h * w)
        shape = target_w, height
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        after_pad = width - target_w
        if after_pad > 0:
            pad_width = [(0, 0), (0, after_pad)]
            # color = get_optimal_pad_value(img)
            color = (0, 0, 0)
            img = [np.pad(img[..., c], pad_width, 'constant', constant_values=color[c]) for c in range(3)]
            img = np.stack(img, -1)
        else:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        assert img.shape == (height, width, 3)
        if debug:
            img_utils.show_img(img, (4, 1))
        else:
            jpeg_utils.write_jpeg(save_img_path, img)


def get_optimal_pad_value(img: np.ndarray) -> tuple:
    h, w, _ = img.shape
    colors = np.reshape(img, (h * w, 3))
    thr_percent = .4
    thr = (np.max(colors, 0) * thr_percent).astype(int_dtype)
    ind = np.array(list(map(np.all, colors > thr)))
    if sum(ind):
        colors = colors[ind]
        color = tuple(np.mean(colors, 0).astype('uint8'))
        return color
    return 0, 0, 0


def remove_old_imgs(dataset_dir_path: str, df_name: str,
                    old_df_name: str, img_dir_pth: str, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    img_dir_pth = Path(img_dir_pth)
    old_df_path = dataset_dir_path / (DF_FILE_FORMAT % old_df_name)
    old_df = pandas_utils.read_dataframe(old_df_path)
    new_df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    new_df = pandas_utils.read_dataframe(new_df_path)
    old_df_index = old_df.index
    new_df_index = new_df.index
    count = 0
    for tag_id in tqdm(old_df_index, total=len(old_df_index), smoothing=.01):
        if tag_id not in new_df_index:
            img_path = get_file_sharding(img_dir_pth, tag_id)
            if img_path.exists():
                count += 1
                if not debug:
                    img_path.unlink()
    if debug:
        print("%s images need to be deleted" % count)
    else:
        print("%s images were deleted" % count)
