from itertools import islice
from pathlib import Path

from pylibs import img_utils
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from pylibs.numpy_utils import read_array
from .data_generation import BatchGenerator
from .training import get_transforms

WHITE_PIXEL = 255


def main(df_name, img_file, filter_fun):
    transform_list = get_transforms()
    dataset_dir_path = Path('../dataset')
    images = read_array(dataset_dir_path / img_file)
    batch_size = 10
    max_text_len = 11

    val_set_path = dataset_dir_path / 'dataframes' / (DF_FILE_FORMAT % df_name)
    val_df = read_dataframe(val_set_path)
    val_df = val_df.iloc[[i for i, x in enumerate(val_df.text) if filter_fun(x)]]
    val_gen = BatchGenerator(val_df, batch_size, images, max_text_len, transform_list, seed=42,
                             augment_prob=1)
    print(val_df.loc[val_df.text == '16162'])
    for batch_x, _ in islice(val_gen, 10):
        for img, label in zip(batch_x['input'], batch_x['texts']):
            if filter_fun(label):
                img_utils.show_img((img + 1) / 2, convert2rgb=False)
                print(label)
