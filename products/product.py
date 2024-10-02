from abc import abstractmethod, ABC
from numpy.typing import NDArray
from numpy import float_
from typing import Union, Tuple

from models.model import Model


class Product(ABC):
    """
    An abstract class for a generic derivative product that can be priced by an object of type Model.
    """
    @abstractmethod
    def get_price(
        self,
        model: Model,
        method: str,
        F0: float,
        **kwargs
    ) -> Union[float, NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        A generic method that calculates the product price in the given model with the given method.

        :param model: a model used for pricing.
        :param method: a method used to price the product.
        :param F0: initial value of the underlying price.
        :return: a price of an array of prices of the product.
        """
        raise NotImplementedError()
