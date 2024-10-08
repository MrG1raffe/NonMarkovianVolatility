import numpy as np
from numpy.typing import NDArray
from numpy import float_


def price_from_volatility(
    vol_proc: NDArray[float_],
    mode: str = 'vol',
    T: float = 1,
    x0: float = 1,
    rho: float = 0,
    dW: NDArray[float_] = None
) -> NDArray[float_]:
    """
    Sample process X (with stochastic volatility) on [0, T], satisfying SDE
    dS = sigma*X*dW,
    given the trajectory of stochastic volatility sigma.

    Args:
        vol_proc: trajectory of the volatility process (on the uniform grid on [0, 1]).
        mode: 'vol' or 'var', whether the vol_proc is volatility or variance process.
        T: time segment size.
        x0: initial value of X.
        rho: correlation between price and volatility driving BMs.
        dW: increments of volatility friving BM. len(dW) = len(vol_process) - 1.

    Returns:
        Trajectory of X on the uniform grid of shape (vol_process.size + 1,).
    """
    vol_proc = vol_proc[:-1]
    dt = T / vol_proc.size
    if dW is None:
        dW = np.random.randn(vol_proc.size) * np.sqrt(dt)
    dB = rho * dW + np.sqrt(1 - rho**2) * np.random.randn(dW.size) * np.sqrt(dt)
    if mode == 'vol':
        return np.exp(np.log(x0) + np.concatenate([[0], vol_proc * dB - 0.5 * vol_proc**2 * dt]).cumsum())
    else:
        return np.exp(np.log(x0) + np.concatenate([[0], np.sqrt(vol_proc) * dB - 0.5 * vol_proc * dt]).cumsum())
