from pathlib import Path

import numpy as np

from pylibs.jpeg_utils import get_shape


def shape_analysis(ds_dir: Path, img_dir: str, ds_set: str):
    images_dir = ds_dir / img_dir / ds_set / 'X'

    widths = []
    heights = []
    for img_path in images_dir.iterdir():
        w, h = get_shape(img_path)
        widths.append(w)
        heights.append(h)
    return np.array(widths), np.array(heights)
