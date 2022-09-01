from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import numpy as np
import turbojpeg
from PIL import Image

from .types import Pathlike

try:
    lib_path = '/home/khan/Programms/miniconda3/envs/main/lib/libturbojpeg.so.0.2.0'
    jpeg = turbojpeg.TurboJPEG(lib_path)
except OSError as e:
    import warnings

    warnings.warn("Could not find libjpeg-turbo, jpeg_utils won't work")
    jpeg = None

color_space2pixel_format_dict = dict(
        rgb=turbojpeg.TJPF_RGB,
        gray=turbojpeg.TJPF_GRAY
)

JPEG_FILENAME_FORMAT = '%s.jpg'
JPEG_SUFFIX = '.jpg'


def read_jpeg(image_path: Pathlike,
              _file: Optional[BinaryIO] = None,
              color_space: str = 'rgb') -> np.ndarray:
    if color_space in color_space2pixel_format_dict:
        pixel_format = color_space2pixel_format_dict[color_space]
    else:
        raise ValueError('No such colorspace: %s' % color_space)
    if isinstance(image_path, Path):
        image_path = str(image_path)
    open_mode = 'rb'
    if _file is None:
        with open(image_path, open_mode) as _file:
            return jpeg.decode(_file.read(), pixel_format=pixel_format)
    else:
        _file.seek(0)  # reset cursor of file
        return jpeg.decode(_file.read(), pixel_format=pixel_format)


def write_jpeg(image_path: Pathlike,
               img: np.ndarray, quality: int = 100):
    if isinstance(image_path, Path):
        image_path = str(image_path)
    open_mode = 'wb'
    with open(image_path, open_mode) as _file:
        _file.write(jpeg.encode(img, quality=quality))


def get_shape(img_path: Pathlike) -> Tuple[int, int]:
    return Image.open(img_path).size
