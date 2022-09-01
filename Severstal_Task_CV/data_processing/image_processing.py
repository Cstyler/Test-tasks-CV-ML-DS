from pathlib import Path

import albumentations as albu
import cv2
import numpy as np

from pylibs import img_utils
from pylibs.numpy_utils import write_array
from pylibs.img_utils import apply_mask


def create_img_array(dataset_dir_path: str, img_dir: str, ds_name,
                     width: int, height: int, save_arr_name: str, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    img_dir = dataset_dir_path / img_dir
    imgs = []
    masks = []

    images_dir = dataset_dir_path / img_dir / ds_name / 'X'
    mask_dir = images_dir.parent / 'y'
    preprocess = albu.Resize(height, width, cv2.INTER_CUBIC, always_apply=True)
    CLASS_VALUES = [0, 1, 2, 3]

    for img_path, mask_path in zip(images_dir.iterdir(), mask_dir.iterdir()):
        try:
            img = cv2.imread(str(img_path))
            try:
                mask = cv2.imread(str(mask_path))
                res = preprocess(image=img, mask=mask)
                img = res['image']
                mask = res['mask']
                assert np.all(mask[..., 0] == mask[..., 1])
                assert np.all(mask[..., 1] == mask[..., 2])
                mask = mask[..., 0]
                mask = np.stack([mask == v for v in CLASS_VALUES], axis=-1).astype('uint8')
                masks.append(mask)
                imgs.append(img)

                if debug:
                    figsize = (15, 10)
                    alpha = .9
                    img_show = apply_mask(img, mask[..., 0], (255, 0, 0), alpha)
                    img_utils.show_img(img_show, figsize)
                    img_show = apply_mask(img, mask[..., 1], (0, 255, 0), alpha)
                    img_utils.show_img(img_show, figsize)
                    img_show = apply_mask(img, mask[..., 2], (0, 0, 255), alpha)
                    img_utils.show_img(img_show, figsize)
                    img_show = apply_mask(img, mask[..., 3], (128, 128, 255), alpha)
                    img_utils.show_img(img_show, figsize)
            except (OSError, IOError):
                print(str(mask_path))
        except (OSError, IOError):
            print(str(img_path))

    imgs_arr = np.asarray(imgs)
    save_path_array = dataset_dir_path / ('%s.npy' % save_arr_name)
    write_array(save_path_array, imgs_arr)
    masks_arr = np.asarray(masks)
    save_path_array = dataset_dir_path / ('%s_masks.npy' % save_arr_name)
    write_array(save_path_array, masks_arr)
