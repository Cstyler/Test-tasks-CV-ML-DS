from Bio.pairwise2 import align
from typing import Tuple
GAP = '-'


def levenshtein_distance(a: str, b: str) -> int:
    """
    calculates levenshtein distance with all costs set to 1
    (see https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm)
    :param a: target string for assessment
    :param b: source string (ground-truth)
    :return: score
    """
    return levenshtein_distance_weighted(a, b, 1, 1, 1)


def levenshtein_distance_weighted(a: str, b: str, add_cost: int, re_cost: int, del_cost: int) -> int:
    """
    calculates levenshtein distance using bio-informatics algorithm
    (see https://en.wikipedia.org/wiki/Needleman–Wunsch_algorithm)
    :param a: target string for assessment
    :param b: source string (ground-truth)
    :param add_cost: cost of adding one symbol
    :param re_cost: cost of replacement of one symbol
    :param del_cost: cost of deletion of one symbol
    :return: non-normalized score
    """
    len_a, len_b = len(a), len(b)
    if not a:
        return len_b * del_cost
    if not b:
        return len_a * add_cost
    a = a.upper()
    b = b.upper()
    a, b = utf8_to_ascii(a, b)

    alignment, *_ = align.globalms(a, b, 0, -2, -2, -2)
    a, b, _, _, _ = alignment
    score = 0
    for target_char, source_char in zip(a, b):
        if target_char != source_char:
            if target_char == GAP and source_char != GAP:
                score += del_cost
            elif source_char == GAP and target_char != GAP:
                score += add_cost
            else:
                score += re_cost
    return score


def utf8_to_ascii(a: str, b: str) -> Tuple[str, str]:
    xs = set(a)
    ys = set(b)
    union = xs | ys
    trans_table = {unicode_symbol: chr(i) for i, unicode_symbol in enumerate(union)}
    trans_table[GAP] = chr(len(union))

    def translate(s: str) -> str:
        return ''.join(map(trans_table.__getitem__, s))

    a = translate(a)
    b = translate(b)
    return a, b

