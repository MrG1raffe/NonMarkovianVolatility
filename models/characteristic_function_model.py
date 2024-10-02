from abc import abstractmethod
from numpy.typing import NDArray
from numpy import complex_
from dataclasses import dataclass

from models.model import Model


@dataclass
class CharacteristicFunctionModel(Model):
    """
    An abstract class to price options in the models with the characteristic function available.
    """

    @abstractmethod
    def characteristic_function(
        self,
        T: float,
        x: float,
        u1: NDArray[complex_],
        **kwargs
    ) -> complex_:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * X_T + i * u2 * V_T + i * f1 * ∫ X_s ds + i * f2 * ∫ V_s ds}]     (1)

        for the given model, where X_t = log(F_t).

        :param u1: X_T coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: X_0, equals to log(F_0).
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        raise NotImplementedError()
