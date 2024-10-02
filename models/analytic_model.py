from abc import abstractmethod
from typing import Union
from numpy.typing import NDArray
from numpy import float_

from models.model import Model


class AnalyticModel(Model):
    """
    A generic class that admits for analytic vanilla pricing formulas.
    """
    @abstractmethod
    def get_vanilla_option_price_analytic(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F0: float,
        flag: str = "call",
        is_vol_surface: bool = False
    ):
        """
        Calculates analytically the prices of the European vanilla with the explicit formula.

        :param T: option maturities, a number or a 1-dimensional array.
        :param K: options strikes. Either a number, or a 1D array of strikes, or a 2D array of shape
            (len(T), len(strikes)) containing in the i-th raw the strikes corresponding to maturity T[i].
        :param F0: initial value of the underlying price.
        :param flag: determines the option type: "c" or "call" for calls, "p" or "put" for puts.
        :param is_vol_surface: whether to return the Black implied volatility value instead of option prices.
        :return: an array of shape (T.size, K.shape[-1]) with the option prices or implied vols.
        """
        raise NotImplementedError
