from abc import ABC
from dataclasses import dataclass


@dataclass
class Model(ABC):
    """
    A generic class for the pricing models.
    """
    def __post_init__(self) -> None:
        """
        Checks the correctness of the given parameters, does necessary pre-calculations.

        :return: None
        """
        pass

    def update_params(
        self,
        new_params: dict
    ) -> None:
        """
        Updates the attributes of the class instance and runs post_init after.

        :param new_params: a dictionary with new parameters values. Its keys are attributes names, the values are
            new attribute values.
        :return: None
        """
        for param_name in new_params:
            if param_name not in self.__dict__:
                raise KeyError(f"Wrong parameter name {param_name} was given to the model.")
            self.__dict__[param_name] = new_params[param_name]
        self.__post_init__()
