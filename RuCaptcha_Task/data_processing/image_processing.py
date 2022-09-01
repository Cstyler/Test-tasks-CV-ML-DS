from pathlib import Path

import cv2
from tqdm import tqdm_notebook as tqdm

from pylibs import img_utils, jpeg_utils, pandas_utils
from pylibs.jpeg_utils import JPEG_FILENAME_FORMAT
from pylibs.pandas_utils import DF_FILE_FORMAT


def resize_images(dataset_dir_path: str, df_name: str, img_dir: str, save_img_dir: str,
                  width: int, height: int, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    source_img_dir = dataset_dir_path / img_dir
    img_resized_dir_path = dataset_dir_path / save_img_dir
    img_resized_dir_path.mkdir(exist_ok=True)
    df_path = dataset_dir_path / 'dataframes' / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    shape = (width, height)
    index = df.index
    for img_id in tqdm(index, total=len(index), smoothing=.01):
        img_path = source_img_dir / (JPEG_FILENAME_FORMAT % img_id)
        save_img_path = img_resized_dir_path / (JPEG_FILENAME_FORMAT % img_id)
        if save_img_path.exists():
            continue
        try:
            img = jpeg_utils.read_jpeg(img_path)
        except (OSError, IOError):
            continue
        img = img_utils.change_color_space(cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC))
        if debug:
            img_utils.show_img(img, (3, 5))
        else:
            jpeg_utils.write_jpeg(save_img_path, img)
