from itertools import islice

from pylibs import img_utils
from .data_generation import BatchGenerator
from .training import get_transforms
from .utils import DATASET_DIR, ARR_FILE_FORMAT

WHITE_PIXEL = 255

def main(img_arr_name, height, width):
    transform_list = get_transforms(height, width)

    img_arr_path = DATASET_DIR / (ARR_FILE_FORMAT % img_arr_name)
    mask_arr_path = DATASET_DIR / (ARR_FILE_FORMAT % (img_arr_name + '_masks'))
    augment_prob = 1.
    batch_size = 10
    val_gen = BatchGenerator(img_arr_path, mask_arr_path, batch_size,
                             transform_list, augment_prob, False, seed=42)

    for batch_x, batch_y in islice(val_gen, 10):
        for img, mask in zip(batch_x, batch_y):
            figsize = (15, 10)
            alpha = .9
            img_show = img_utils.apply_mask(img, mask[..., 0], (255, 0, 0), alpha)
            img_utils.show_img(img_show, figsize)
            img_show = img_utils.apply_mask(img, mask[..., 1], (0, 255, 0), alpha)
            img_utils.show_img(img_show, figsize)
            img_show = img_utils.apply_mask(img, mask[..., 2], (0, 0, 255), alpha)
            img_utils.show_img(img_show, figsize)
            img_show = img_utils.apply_mask(img, mask[..., 3], (128, 128, 255), alpha)
            img_utils.show_img(img_show, figsize)
