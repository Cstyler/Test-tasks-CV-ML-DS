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

def apply_mask(img: np.ndarray, mask: np.ndarray, color, alpha: float =0.5):
    """Apply the given mask to the image.
    """
    img = img.copy()
    eq_mask = mask == 1
    for c in range(3):
        img_c = img[:, :, c]
        img[:, :, c] = np.where(eq_mask, img_c * (1 - alpha) + alpha * color[c] * 255, img_c)
    return img
