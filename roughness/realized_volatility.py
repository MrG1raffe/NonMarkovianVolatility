import numpy as np
from numpy.typing import NDArray
from numpy import float_


def realized_volatility(
    X: NDArray[float_],
    n_est: int = 100,
    T: float = 1,
    mode: str = 'vol'
) -> NDArray[float_]:
    """
    Calculates a realized volatility (RV) estimation for X using n_est points of X at every node.
    RV is calculated via the formula (sum over N_est points in the group corresponding to t)
    RV_t = sqrt(sum(log(S_{i+1}) - log(S_{i}))).
    If aggregate=True, uses each group of n_est points to obtain one point of RV process, otherwise calculates
    moving RV with window size equal to n_est.

    Args:
        X: trajectory of the process.
        n_est: number of points in a cluster.
        T: time segment size.
        mode: 'vol' or 'var', whether the rv is volatility or variance process.

    Returns:
        Trajectory of RV process.
    """
    delta = T / (X.size / n_est)
    n_clusters = (X.size - 1) // n_est
    n_est = (X.size - 1) // n_clusters
    # truncate X to have exactly n_clusters groups of size n_est.
    X = X[:(n_est*n_clusters + 1)]
    ids = np.repeat(range(n_clusters), n_est)
    var = np.bincount(ids, np.diff(np.log(X))**2) / delta
    return var if mode == 'var' else np.sqrt(var)
