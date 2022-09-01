from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cytoolz
import numpy as np
import pandas as pd
import tqdm
import ujson as json
from pylibs.rect_utils import UniversalRect
from pylibs.storage_utils import gen_unique_id, get_file_sharding, get_tag_from_share, save_file_sharding

from pylibs import cutter, img_utils, pandas_utils
from pylibs.jpeg_utils import read_jpeg, write_jpeg
from pylibs.pandas_utils import DF_FILE_FORMAT

INDEX_NAME = 'seg_id'
SEED = 42
np.random.seed(SEED)


def process(dataset_dir_path: str, df_name: str, processed_df_name: str, img_dir: str,
            w_padding: float, h_padding: float,
            debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    processed_images_dir_path = dataset_dir_path / img_dir

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    processed_df = process_df(df_path, processed_images_dir_path, w_padding, h_padding, debug)
    df_save_path = dataset_dir_path / (DF_FILE_FORMAT % processed_df_name)
    if not debug:
        pandas_utils.write_dataframe(processed_df, df_save_path)


def process_df(df_path: Path, img_dir: Path,
               w_padding: float, h_padding: float, debug: bool):
    rows = []
    df = pandas_utils.read_dataframe(df_path)
    id_len = 12
    dataframe_iterator = tqdm.tqdm_notebook(df.iterrows(), total=len(df.index), smoothing=.01)
    for seg_id, row in dataframe_iterator:
        values = json.loads(row['values'])
        segments = json.loads(row['segments'])
        seg_names = tuple(values.keys() & segments.keys())
        if not seg_names:
            continue
        photo_id = row['photo.id']
        img_path = get_tag_from_share(seg_id, photo_id)
        img = read_jpeg(img_path)
        market_id = row['photo.metaInfo.marketId']
        for seg_name in seg_names:
            seg_id = gen_unique_id(id_len)
            seg_path = save_file_sharding(img_dir, Path(seg_id))
            segment = segments[seg_name]
            try:
                unirect = UniversalRect.from_coords_dict(segment)
                unirect = unirect.add_padding(w_padding, h_padding)
                segment_img = cutter.cut_segment_from_img(img, unirect)
                write_jpeg(seg_path, segment_img)
            except Exception as e:
                print(e)
                continue
            text = str(values[seg_name])
            row = (seg_id, text, seg_id, market_id, seg_name)
            rows.append(row)
            if debug:
                img_utils.show_img(img)
                img_utils.show_img(segment_img)
                print(unirect)
                print(text)

    df = pd.DataFrame(rows, columns=(INDEX_NAME, 'text', 'tag_id', 'market_id', 'seg_type'))
    df.set_index(INDEX_NAME, inplace=True)
    return df


KOP_NAMES = frozenset(('CardKop', 'DiscountKop', 'RegularKop'))


def double_zeros_for_kops(row: pd.Series, columns: Iterable[str]) -> list:
    text = row['text']
    if text == '0' and row['seg_type'] in KOP_NAMES:
        text = '00'
    return [text if col == 'text' else row[col] for col in columns]


def add_zero_at_begin_modify_fun(row: pd.Series, columns: Iterable[str]) -> list:
    text = row['text']
    if len(text) == 1:
        text = '0' + text
    return [text if col == 'text' else row[col] for col in columns]


def modify_df_rows(dataset_dir_path: str, df_name: str, new_df_name: str,
                   modify_fun: Callable[[pd.Series, Iterable[str]], list]):
    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    new_df = modify_df_rows_helper(df, modify_fun)
    new_df_path = df_path.with_name(DF_FILE_FORMAT % new_df_name)
    pandas_utils.write_dataframe(new_df, new_df_path)


def modify_df_rows_helper(df, modify_fun):
    rows = []
    for id_, row in df.iterrows():
        new_row = [id_] + modify_fun(row, df.columns)
        rows.append(new_row)
    new_df = pd.DataFrame(rows, columns=[INDEX_NAME] + list(df.columns))
    new_df.set_index(INDEX_NAME, inplace=True)
    return new_df


def post_process_labels_model_grcnn(dataset_dir_path: str, df_names: Iterable[Tuple[str, str]], max_len: int,
                                    outlier_len: int, chars: List[str]):
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
            if len_vals > outlier_len:
                continue
            feature = extend_blank(map(chars2num.__getitem__, text_list), len_vals)
            new_row = seg_id, json.dumps(feature), len_vals, "".join(text)
            rows.append(new_row)
        new_df = pd.DataFrame(rows, columns=(INDEX_NAME, 'label', 'label_len', 'text'))
        if df_name:
            df_path = df_path.with_name(df_name)
        new_df.set_index(INDEX_NAME, inplace=True)
        pandas_utils.write_dataframe(new_df, df_path)

    dataset_dir_path = Path(dataset_dir_path)
    for df_name, save_df_name in df_names:
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


def update_dataset_with_google_annotations(dataset_dir_path: str, df_name: str, save_df_name: str,
                                           divide_size: int = 14, img_dir="processed_images"):
    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    images_dir_path = dataset_dir_path / img_dir
    df = pandas_utils.read_dataframe(df_path)
    from pylibs.google_recognizer import Recognizer
    recognizer = Recognizer()

    src_type = 'file'
    new_rows = []

    google_text_col = 'google_text'
    if google_text_col in df:
        df.drop(google_text_col, axis=1, inplace=True)
    for i, batch in enumerate(cytoolz.partition_all(divide_size, df.iterrows())):
        sources = (get_file_sharding(images_dir_path, seg_id) for seg_id, _ in batch)
        results = recognizer.recognize_concrete_data(sources, src_type, 'code')
        for (seg_id, row), (google_text, _) in zip(batch, results):
            new_row = (seg_id,) + tuple(row) + (google_text,)
            new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows, columns=(INDEX_NAME,) + tuple(df.columns) + (google_text_col,))
    new_df.set_index(INDEX_NAME, inplace=True)

    df_path = df_path.with_name(DF_FILE_FORMAT % save_df_name)
    pandas_utils.write_dataframe(new_df, df_path)


def add_column_to_df(dataset_dir_path: str, df_name: str, df_name2: str, key: str):
    dataset_dir_path = Path(dataset_dir_path)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df1 = pandas_utils.read_dataframe(df_path)
    df1 = pd.DataFrame(df1.loc[:, key])

    df_path2 = dataset_dir_path / (DF_FILE_FORMAT % df_name2)
    df2 = pandas_utils.read_dataframe(df_path2)

    new_df = df2.merge(df1, on=INDEX_NAME)
    pandas_utils.write_dataframe(new_df, df_path2)


def rename_index(df: pd.DataFrame, *,
                 index_name=INDEX_NAME) -> pd.DataFrame:
    return df.index.rename(index_name)


def rename_column(df: pd.DataFrame, *,
                  col_name: str, new_col_name: str) -> pd.DataFrame:
    return df.rename(columns={col_name: new_col_name})


def drop_column(df: pd.DataFrame, *,
                col_name: str) -> pd.DataFrame:
    return df.drop(columns=col_name)


def modify_df(dataset_dir_path: str, df_name: str, modify_fun: Callable[[pd.DataFrame], pd.DataFrame],
              save_df_name: Optional[str] = None):
    dataset_dir_path = Path(dataset_dir_path)
    new_df_name = DF_FILE_FORMAT % (save_df_name if save_df_name else df_name)
    df_name = DF_FILE_FORMAT % df_name
    df_path = dataset_dir_path / df_name
    df = pandas_utils.read_dataframe(df_path)
    df = modify_fun(df)
    pandas_utils.write_dataframe(df, dataset_dir_path / new_df_name)


def subsample_dataset(dataset_dir_path: str, df_name: str, num: int, save_df_name: str):
    def sample(df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(num, random_state=SEED)

    modify_df(dataset_dir_path, df_name, sample, save_df_name)


def subsample_relabel(dataset_dir_path: str, df_name: str, save_df_name: str, num: int):
    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    new_df_path = df_path.with_name(DF_FILE_FORMAT % save_df_name)
    df = pandas_utils.read_dataframe(df_path)
    half_num = num // 2
    len1_df = df.iloc[[i for i, x in enumerate(df.text) if len(x) == 1 and x != '0']].sample(half_num)
    zero_df: pd.DataFrame = df.iloc[[i for i, x in enumerate(df.text) if x == '0']].sample(half_num)
    zero_df = modify_df_rows_helper(zero_df, double_zeros_for_kops)
    len1_df = modify_df_rows_helper(len1_df, add_zero_at_begin_modify_fun)
    df_list = [zero_df, len1_df]
    df_list.extend(df.iloc[[i for i, x in enumerate(df.text) if len(x) == n]] for n in range(2, 6))
    df_list = [df.sample(num if len(df.index) > num else len(df.index), random_state=SEED) for df in df_list]
    new_df = pd.concat(df_list)
    pandas_utils.write_dataframe(new_df, new_df_path)


def reorder_dataframe(dataset_dir_path: str, df1_name: str, df2_name: str):
    dataset_dir_path = Path(dataset_dir_path)
    df1_path = dataset_dir_path / (DF_FILE_FORMAT % df1_name)
    df1 = pandas_utils.read_dataframe(df1_path)
    df2_path = dataset_dir_path / (DF_FILE_FORMAT % df2_name)
    df2 = pandas_utils.read_dataframe(df2_path)
    df2 = df2.loc[[i for i in df1.index if i in df2.index]]
    pandas_utils.write_dataframe(df2, df2_path)
