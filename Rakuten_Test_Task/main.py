from typing import List

def solution(_list: List[int]) -> int:
    p, r = False, False
    max_pit_len = -1
    cur_pit_len = 1
    for i in range(len(_list) - 1):
        cur = _list[i]
        next = _list[i + 1]
        if cur > next:
            if not p and not r:
                p = True
                cur_pit_len += 1
            elif p and not r:
                cur_pit_len += 1
            elif p and r:
                if cur_pit_len > max_pit_len:
                    max_pit_len = cur_pit_len
                cur_pit_len = 1
                p, r = False, False
        elif cur < next:
            if p:
                r = True
                cur_pit_len += 1
        else:
            if cur_pit_len > max_pit_len:
                max_pit_len = cur_pit_len
            cur_pit_len = 1
            p, r = False, False
    if cur_pit_len > max_pit_len and p and r:
        return cur_pit_len

    return max_pit_len


def main():
    import numpy as np
    tests1 = [
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
    low, high = -10000000, 10000000
    tests2 = [
        list(np.random.randint(low, high, size=1000000).astype('int')),
        list(np.random.randint(low, high, size=1000000).astype('int')),
        list(np.random.randint(low, high, size=1000000).astype('int'))
    ]
    for test in tests2:
        print(solution(test))


if __name__ == '__main__':
    main()
