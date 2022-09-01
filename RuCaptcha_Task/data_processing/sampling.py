from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from pylibs import pandas_utils
from pylibs.pandas_utils import DF_FILE_FORMAT

SEED = 42

np.random.seed(SEED)


def train_val_test_split(df: pd.DataFrame, val_size: float, test_size: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_len = len(df.index)
    val_index = int(df_len * (1 - (val_size + test_size)))
    test_index = int(df_len * (1 - test_size))
    shuffled_df = df.sample(frac=1)
    return np.split(shuffled_df, (val_index, test_index))


def split_dataset(dataset_dir_path: str, split_df_name: str, val_size: float, test_size: float,
                  df_name1: str, df_name2: str, df_name3: str):
    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % split_df_name)
    df = pandas_utils.read_dataframe(df_path)
    print("DF size:", len(df.index))
    x1, x2, x3 = train_val_test_split(df, val_size, test_size)
    set_path1 = dataset_dir_path / (DF_FILE_FORMAT % df_name1)
    print("size1:", len(x1.index))
    pandas_utils.write_dataframe(x1, set_path1, index=True)
    set_path2 = dataset_dir_path / (DF_FILE_FORMAT % df_name2)
    print("size2:", len(x2.index))
    pandas_utils.write_dataframe(x2, set_path2, index=True)
    set_path3 = dataset_dir_path / (DF_FILE_FORMAT % df_name3)
    print("size3:", len(x3.index))
    pandas_utils.write_dataframe(x3, set_path3, index=True)
