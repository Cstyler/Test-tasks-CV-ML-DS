from typing import List
import numpy as np


def solution(a: List[int]) -> int:
    def gen_partial_sums():
        yield 0
        _sum = 0
        for i in range(len(a)):
            _sum += a[i]
            yield _sum

    partial_sums = sorted(gen_partial_sums())
    min_abs_slice = min(abs(partial_sums[i] - partial_sums[i - 1]) for i in range(1, len(a) + 1))
    return min_abs_slice


def main():
    tests1 = [
        [2, -4, 6, -3, 9],
        [10, 20, 30, -2, 5, 10],
        [1, 2, 3, -1, 0, 1, 12, 3, 4],
        [0, 1, 3, -2, 0, 1, 0, -3, 2, 3],
        [0, 1, 3, -2, 0, 1, 2, 0, -3, 2, 3],
        [0, 1, 3, -2, 0, 1, 2, 3, 0, -3, 2, 3],
        [0, 1, 3, -2, 0, 1, 3, 0, -3, 2, 3],
        [0, 0, -1, 0, 1, 1, 1, 3, 4, 6],
        [0, -1, 0, 1, 1],
        [0, -1, 0, 1],
        [-1, 0, -1, 0, 1],
        [1, 2, 3, 100, 59, 20, 30, 200],
        [1, 2, 3, 100, 59, 20, 30, 20, 10, 0],
        [1, 2, 3, 4]
    ]
    low, high = -10000, 10000
    size = 10**4
    tests2 = [list(np.random.randint(low, high, size=size).astype('int')) for _ in range(100)]
    for test in tests2:
        # print(test)
        # print(min(test), max(test))
        print(solution(test))


if __name__ == '__main__':
    main()
