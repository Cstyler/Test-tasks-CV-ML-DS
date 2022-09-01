import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import albumentations as albu
import numpy as np
from tensorflow.keras.preprocessing import image

from pylibs.numpy_utils import read_array


class BaseGenerator(image.Iterator):
    pass


class BatchGenerator(BaseGenerator):
    def __init__(self, img_arr_path: Path, mask_arr_path: Path,
                 batch_size: int,
                 transform_list: Optional[List[albu.BasicTransform]] = None,
                 augment_prob: float = 0.5, apply_normalization: bool = True,
                 shuffle: bool = True, seed: Optional[int] = None):
        self.images = read_array(img_arr_path)
        self.masks = read_array(mask_arr_path)
        self.apply_augmentation = transform_list is not None
        self.normalize_trans = albu.Normalize(always_apply=True) if apply_normalization else albu.NoOp()
        if self.apply_augmentation:
            self.augmentations = albu.Compose(transform_list, p=augment_prob)
        n = self.images.shape[0]
        super().__init__(n, batch_size, shuffle, seed)

    @property
    def steps_per_epoch(self):
        return math.ceil(self.n / self.batch_size)

    @staticmethod
    def normalize_img2(img: np.ndarray) -> np.ndarray:
        img = 2 * (img / 255) - 1
        return img.astype(np.float32, copy=False)

    def normalize_img(self, img: np.ndarray) -> np.ndarray:
        return self.normalize_trans(image=img)['image']

    def iter_images(self, index_array: np.ndarray) -> Iterable[np.ndarray]:
        for img_ind in index_array:
            img = self.images[img_ind]
            mask = self.masks[img_ind]
            img, mask = self.augment_img(img, mask)
            img = self.normalize_img(img)
            yield img, mask.astype('float32', copy=False)

    def augment_img(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.apply_augmentation:
            aug_res = self.augmentations(image=img, mask=mask)
            return aug_res['image'], aug_res['mask']
        return img, mask

    def _get_batches_of_transformed_samples(self,
                                            index_array: np.ndarray) \
            -> Tuple[np.ndarray, ...]:
        """Gets a batch of transformed samples.
        # Arguments
            index_array: array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """

        # transpose list of lists
        return tuple(np.stack(samples) for samples in zip(*self.iter_images(index_array)))
