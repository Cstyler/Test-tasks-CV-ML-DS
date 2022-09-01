import numpy as np

from .types import Pathlike


def print_stats(a: np.ndarray, ndigits: int = 5):
    print("Mean:", round(np.mean(a), ndigits), ", Median:",
          round(np.median(a), ndigits), ", Min:",
          round(np.min(a), ndigits), ", Max:", round(np.max(a), ndigits))

NPY_SUFFIX = '.npy'


def write_array(path: Pathlike, array: np.ndarray):
    if not isinstance(path, str):
        path = str(path)
    with open(path, 'wb') as file_:
        np.save(file_, array)


def read_array(path: Pathlike):
    if not isinstance(path, str):
        path = str(path)
    with open(path, 'rb') as file_:
        return np.load(file_)
