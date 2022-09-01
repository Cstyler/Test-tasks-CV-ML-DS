from pathlib import Path

import numpy as np
from tqdm import tqdm_notebook as tqdm

from pylibs.jpeg_utils import get_shape, JPEG_FILENAME_FORMAT
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe


def shape_analysis(ds_dir: Path, df_name: str, img_dir: str):
    df_path = ds_dir / 'dataframes' / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)
    images_dir = ds_dir / img_dir

    widths = []
    heights = []
    for img_id, row in tqdm(df.iterrows(), total=len(df.index), smoothing=.01):
        img_path = images_dir / (JPEG_FILENAME_FORMAT % img_id)
        w, h = get_shape(img_path)
        widths.append(w)
        heights.append(h)
    return np.array(widths), np.array(heights)
