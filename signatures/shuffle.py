import numpy as np
from numpy import int64
from numba import jit
from typing import Tuple

from signatures.numba_utility import combinations


@jit(nopython=True)
def shuffle_product(word_1: int, word_2: int) -> Tuple:
    """
    Computes the shuffle product of two words, resulting in a dictionary where the keys
    are the possible shuffles (as int) of the two words, and the values are the counts of each shuffle.

    The shuffle product interleaves the letters of `word_1` and `word_2` in all possible ways while maintaining
    the relative order of letters within each word.

    :param word_1: The first word as an integer.
    :param word_2: The second word as an integer.
    :return: A pair of resulting words from the shuffle product and counts of each word.
    """
    if word_1 == 0:
        return np.array([word_2], dtype=int64), np.ones(1, dtype=int64)
    if word_2 == 0:
        return np.array([word_1], dtype=int64), np.ones(1, dtype=int64)

    l1, l2 = (np.log10(np.array([word_1, word_2], dtype=int64)) + 1).astype(int64)
    word_concat = word_1 * 10**l2 + word_2
    letters = np.array([word_concat // 10**k % 10 for k in range(l1 + l2 - 1, -1, -1)])

    indices_left = combinations(np.arange(l1 + l2), l1)
    indices_right = combinations(np.arange(l1 + l2), l2)[::-1]

    indices = np.zeros((indices_left.shape[0], l1 + l2))
    indices[:, :l1] = indices_left
    indices[:, l1:] = indices_right

    powers = l1 + l2 - 1 - indices
    shuffle = np.sum((10 ** powers).astype(int64) * letters.astype(int64), axis=1)

    sorted_shuffle = np.zeros(shuffle.size + 1, dtype=int64)
    sorted_shuffle[:shuffle.size] = np.sort(shuffle)
    sorted_shuffle[shuffle.size] = -1

    change_indices = np.where(np.diff(sorted_shuffle) != 0)[0]
    counts = np.zeros(change_indices.size, dtype=int64)
    counts[0] = change_indices[0] + 1
    counts[1:] = np.diff(change_indices)

    return sorted_shuffle[change_indices], counts
