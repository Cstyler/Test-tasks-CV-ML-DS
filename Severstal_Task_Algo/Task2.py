from random import random as gen_random_number
from typing import List

import numpy as np


def bisect(a: List[float], x: float):
    hi = len(a)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def cumulative_distribution(weights: np.ndarray) -> List[float]:
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum)
    return result


class DataLoader:
    def __init__(self, probs):
        self.probs = probs
        self.cdf_vals = cumulative_distribution(self.probs)

    def get_element(self):
        return bisect(self.cdf_vals, gen_random_number())


def main():
    probs = np.asarray([0, .05, .1, .15, .2, .25, .25])
    n = len(probs)
    dl = DataLoader(probs)

    run_num = 10 ** 6
    results = np.asarray([dl.get_element() for _ in range(run_num)])
    result_freqs = np.asarray([(results == i).sum() for i in range(n)]) / run_num
    diff = (result_freqs - probs) * 100
    accuracy = 0.08
    print("Diff from probs:", np.abs(diff))
    assert np.all(np.abs(diff) < accuracy)


if __name__ == '__main__':
    main()
