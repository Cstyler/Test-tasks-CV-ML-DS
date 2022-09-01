from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from pylibs.numpy_utils import read_array, write_array
from sklearn.model_selection import train_test_split

SEED = 42

np.random.seed(SEED)


def train_val_test_split(df: pd.DataFrame, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_len = len(df.index)



def split_dataset(dataset_dir_path: str, img_arr_name: str, val_size: float):
    dataset_dir_path = Path(dataset_dir_path)

    img_arr_path = dataset_dir_path / ('%s.npy' % img_arr_name)
    imgs_arr = read_array(img_arr_path)

    mask_arr_path = dataset_dir_path / ('%s_masks.npy' % img_arr_name)
    masks_arr = read_array(mask_arr_path)

    x_train, x_val, y_train, y_val = train_test_split(imgs_arr, masks_arr, test_size=val_size, shuffle=True)
    x_train_path = dataset_dir_path / ('%s_train.npy'  % img_arr_name)
    write_array(x_train_path, x_train)
    print("x train shape:", x_train.shape)
    y_train_path = dataset_dir_path / ('%s_train_masks.npy' % img_arr_name)
    write_array(y_train_path, y_train)
    print("y train shape:", y_train.shape)

    x_val_path = dataset_dir_path / ('%s_val.npy' % img_arr_name)
    write_array(x_val_path, x_val)
    print("x val shape:", x_val.shape)
    y_val_path = dataset_dir_path / ('%s_val_masks.npy' % img_arr_name)
    write_array(y_val_path, y_val)
    print("y val shape:", y_val.shape)
