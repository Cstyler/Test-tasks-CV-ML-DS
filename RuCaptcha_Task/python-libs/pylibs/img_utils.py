from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def change_color_space(img: np.ndarray, code: int = cv2.COLOR_BGR2RGB) -> np.ndarray:
    return cv2.cvtColor(img, code)


def show_img(img: np.ndarray, figsize: Optional[Tuple[int, int]] = None, convert2rgb: bool = True):
    if convert2rgb:
        img = change_color_space(img)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
