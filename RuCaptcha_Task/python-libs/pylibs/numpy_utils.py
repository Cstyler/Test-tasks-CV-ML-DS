import numpy as np

from .types import Pathlike


def print_stats(a: np.ndarray, ndigits: int = 5):
    print("Mean:", round(np.mean(a), ndigits), ", Median:",
          round(np.median(a), ndigits), ", Min:",
          round(np.min(a), ndigits), ", Max:", round(np.max(a), ndigits))


def top_k(a: np.ndarray, k: int) -> np.ndarray:
    """
    :param a: 1-d array
    :param k: int, num of samples
    :return: ndarray of top k samples from a
    """
    ind = np.argpartition(a, -k)[-k:]
    ind = ind[np.argsort(a[ind])]
    return ind


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
