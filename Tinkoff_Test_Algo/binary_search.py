from typing import List
 
 
def binary_search(arr: List[int], l: int, r: int, x: int) -> bool:
    """
    :param arr: array in which the search occurs
    :param l: leftmost index of current array
    :param r: rightmost index of current array + 1
    :param x: value to search
    """
    if l == r:
        return arr[l] == x
    m = (l + r) // 2
    val = arr[m]
    if val == x:
        return True
    elif x < val:
        return binary_search(arr, l, m, x)
    else:
        return binary_search(arr, m + 1, r, x)

def test():
    arr = [-1, 1, 2, 5, 10, 23]
    assert binary_search(arr, 1, 3, 2)
    assert not binary_search(arr, 1, 3, 5)
    assert not binary_search(arr, 2, 5, 10)
    assert binary_search(arr, 2, 5, 5)
    assert not binary_search(arr, 2, 5, 23)
    assert binary_search(arr, 0, 1, -1)
    assert binary_search(arr, 0, 6, 1)
 
if __name__ == '__main__':
    pass
