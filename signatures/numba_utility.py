import numpy as np
from numba import jit, int64


@jit(nopython=True)
def factorial(x):
    n = 1
    for i in range(2, x+1):
        n *= i
    return n


@jit(nopython=True)
def combinations(pool, r):
    n = len(pool)
    indices = list(range(r))
    empty = not(n and (0 < r <= n))
    l = (factorial(n) // factorial(r) // factorial(n - r))
    res = np.zeros((l, r), dtype=int64)
    it = 0

    if not empty:
        result = [pool[i] for i in indices]
        res[it] = result
        it = it + 1

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j - 1] + 1

            result = [pool[i] for i in indices]
            res[it] = result
            it = it + 1

    return res