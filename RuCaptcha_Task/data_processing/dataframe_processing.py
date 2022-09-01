from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import ujson as json

from pylibs import pandas_utils, jpeg_utils
from pylibs.numpy_utils import write_array, np
from pylibs.pandas_utils import DF_FILE_FORMAT

INDEX_NAME = 'num'
SEED = 42
np.random.seed(SEED)


def check_text(text: str, alphabet: set):
    for x in text:
        if x not in alphabet:
            return False
    return True


def preprocess_df(dataset_dir_path: str, df_name: str, save_df_name: str, alphabet: set):
    dataset_dir_path = Path(dataset_dir_path)

    DF_FILE_FORMAT = "%s.csv"
    df_name = DF_FILE_FORMAT % df_name
    df = pd.read_csv(dataset_dir_path / df_name)

    df.loc[12318, 'text'] = '77501'
    df.loc[13873, 'text'] = '74517'
    df.loc[14092, 'text'] = '54109'

    rows = []
    for _, row in df.iterrows():
        if check_text(row['text'], alphabet):
            rows.append(row)
    df = pd.DataFrame(rows, columns=df.columns)
    df.set_index(INDEX_NAME, inplace=True)

    save_df_name = pandas_utils.DF_FILE_FORMAT % save_df_name
    save_df_path = dataset_dir_path / save_df_name
    pandas_utils.write_dataframe(df, save_df_path)


def post_process_labels_model_grcnn(dataset_dir_path: str, df_names: Iterable[Tuple[str, str]],
                                    max_len: int, chars: List[str]):
    char_set = set(chars)
    n_classes = len(chars)
    chars2num = {x: i for i, x in enumerate(chars)}

    def extend_blank(iterable: Iterable[int], len_: int) -> List[int]:
        extend_size = max_len - len_
        return list(iterable) + [n_classes] * extend_size

    def process_df(df_path: Path, df_name: Optional[str] = None):
        rows = []
        df = pandas_utils.read_dataframe(df_path)
        for seg_id, row in df.iterrows():
            text = row['text']
            text_list = [x for x in text if x in char_set]
            len_vals = len(text_list)
            feature = extend_blank(map(chars2num.__getitem__, text_list), len_vals)
            new_row = seg_id, json.dumps(feature), "".join(text)
            rows.append(new_row)
        new_df = pd.DataFrame(rows, columns=(INDEX_NAME, 'label', 'text'))
        if df_name:
            df_path = df_path.with_name(df_name)
        new_df.set_index(INDEX_NAME, inplace=True)
        pandas_utils.write_dataframe(new_df, df_path)

    dataset_dir_path = Path(dataset_dir_path)
    df_name, save_df_name = df_names
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    process_df(df_path, DF_FILE_FORMAT % save_df_name)


def post_process_labels_model_freq(dataset_dir_path: str, df_name: str, new_df_name: str):
    chars = tuple(map(str, range(10)))

    def calc_char_freq(s: str) -> List[int]:
        char_count = Counter(s)
        return [char_count[c] for c in chars]

    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    rows = [(seg_id, json.dumps(calc_char_freq(row['text']))) for seg_id, row in df.iterrows()]
    new_df = pd.DataFrame(rows, columns=(INDEX_NAME, 'char_freq'))
    new_df.set_index(INDEX_NAME, inplace=True)
    pandas_utils.write_dataframe(new_df, df_path.with_name(DF_FILE_FORMAT % new_df_name))


def create_img_array(dataset_dir_path: str, df_name: str, img_dir: str, save_arr_name: str, save_df_name: str):
    dataset_dir_path = Path(dataset_dir_path)
    img_dir = dataset_dir_path / img_dir
    df_path = dataset_dir_path / 'dataframes' / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    imgs = []
    rows = []
    for i, img_id in enumerate(df.index):
        img_path = img_dir / (jpeg_utils.JPEG_FILENAME_FORMAT % img_id)
        try:
            imgs.append(jpeg_utils.read_jpeg(str(img_path)))
            rows.append((img_id, i))
        except OSError:
            print(str(img_path))
    df = pd.DataFrame(rows, columns=(INDEX_NAME, 'array_ind'))
    df.set_index(INDEX_NAME, inplace=True)
    save_df_path = dataset_dir_path / 'dataframes' / (DF_FILE_FORMAT % save_df_name)
    pandas_utils.write_dataframe(df, save_df_path)
    arr = np.asarray(imgs)
    save_path_array = dataset_dir_path / ('%s.npy' % save_arr_name)
    write_array(save_path_array, arr)


def add_column_to_df(dataset_dir_path: str, df_name: str, df_name2: str, key: str):
    dataset_dir_path = Path(dataset_dir_path)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df1 = pandas_utils.read_dataframe(df_path)
    df1 = pd.DataFrame(df1.loc[:, key])

    df_path2 = dataset_dir_path / (DF_FILE_FORMAT % df_name2)
    df2 = pandas_utils.read_dataframe(df_path2)

    new_df = df2.merge(df1, on=INDEX_NAME)
    pandas_utils.write_dataframe(new_df, df_path2)
