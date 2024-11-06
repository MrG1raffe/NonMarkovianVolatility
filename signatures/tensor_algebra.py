import numpy as np
from numpy.typing import NDArray
from numpy import float64, int64
from typing import Union
import matplotlib.pyplot as plt

from signatures.tensor_sequence import TensorSequence
from signatures.alphabet import Alphabet
import iisignature


class TensorAlgebra:
    __alphabet: Alphabet

    def __init__(self, dim: int):
        """
        Initializes an instance of tensor algebra that keeps the alphabet and facilitates the use of JitClass
        TensorSequence.

        :param dim: The dimension, representing the number of distinct letters.
        """

        self.__alphabet = Alphabet(dim=dim)
        self.jit_compile()

    def jit_compile(self) -> None:
        """
        Compiles all the jit-classes and jit-functions needed to work with TensorSequence.
        """
        print("Compiling...")
        (self.__alphabet.index_to_word(0), self.__alphabet.index_to_length(np.ones(1, dtype=int64)),
         self.__alphabet.dim, self.__alphabet.word_to_base_dim_number(""), self.__alphabet.word_to_index(""))
        self.from_dict(word_dict={"": 1, "2": 2, "22": 0.5, "12": 3, "112": 4, "2212": 5}, trunc=4)
        ts = self.from_word("", trunc=2)
        ts2 = self.from_dict({"1": 2, "2": 3}, trunc=2)
        self.from_dict(word_dict={"": np.arange(5), "2": 2 * np.arange(5)}, trunc=2)
        self.from_array(trunc=2, array=np.ones((15, 10, 10)))
        empty = self.from_dict({}, trunc=1)
        (ts[""], 2 * ts, ts * 2, ts / 2, ts + ts2, ts - ts, ts @ ts)
        (len(ts), bool(ts), ts.array, ts.trunc, ts.indices)
        (ts.norm_inf(), ts.proj(""), ts.tensor_prod_word("1"), ts.tensor_prod_index(1),
         ts.tensor_prod(ts), ts.shuffle_prod(ts), ts.tensor_pow(1), ts.shuffle_pow(1),
         ts.tensor_exp(1), ts.shuffle_exp(1), (ts / 2).resolvent(1))
        print("Compilation finished.")

    @property
    def alphabet(self) -> Alphabet:
        """
        Returns the alphabet as an Alphabet instance.

        :return: The alphabet object.
        """
        return self.__alphabet

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the alphabet.

        :return: The number of distinct letters in the alphabet.
        """
        return self.__alphabet.dim

    def from_word(self, word: Union[str, int], trunc: int) -> TensorSequence:
        """
        Creates a TensorSequence from a given word and a truncation level.

        :param word: A word to transform into a tensor sequence.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
        :return: A TensorSequence constructed from the provided word and truncation level.
        """
        indices = np.array([self.__alphabet.word_to_index(str(word))])
        array = np.ones(1)

        return TensorSequence(self.alphabet, trunc, array, indices)

    def from_dict(self, word_dict: dict, trunc: int) -> TensorSequence:
        """
        Creates a TensorSequence from a given word dictionary and a truncation level.

        :param word_dict: A dictionary where keys are words (as strings or integers) and values are their coefficients.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
        :return: A TensorSequence constructed from the provided word dictionary and truncation level.
        """
        if not word_dict:
            indices = np.zeros((0,), dtype=int64)
            array = np.zeros((0, 1, 1))
        else:
            # sort the arrays with respect to indices
            indices, array = zip(*sorted(zip(
                [self.__alphabet.word_to_index(str(word)) for word in word_dict.keys()],
                list(word_dict.values())
            )))
            indices = np.array(indices, dtype=int64)
            array = np.array(array)

        return TensorSequence(self.alphabet, trunc, array, indices)

    def to_dict(self, ts: TensorSequence) -> dict:
        """
        Converts the TensorSequence to a dictionary mapping words to their corresponding coefficients.

        :return: A dictionary where keys are words and values are tensor coefficients.
        """
        keys = [self.__alphabet.index_to_word(index) for index in ts.indices]
        values = ts.array.squeeze()
        if np.all(np.isclose(values.imag, 0)):
            values = values.real
        return dict(zip(keys, values))

    def from_array(self, trunc: int, array: NDArray, indices: NDArray = None) -> TensorSequence:
        """
        Creates a TensorSequence from a given array and (optionally) indices. If indices are not given, takes the
        indices of the first elements of the tensor algebra.

        :param array: An array of coefficients.
        :param indices: Indices of words corresponding to the coefficients in the array.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
        :return: A TensorSequence constructed from the provided array and indices.
        """
        array = np.array(array, dtype=float64)
        if indices is None:
            indices = np.arange(array.shape[0])

        return TensorSequence(self.alphabet, trunc, array, indices)

    def to_str(self, ts: TensorSequence) -> str:
        """
        Converts an instance of TensorSequence into a string.

        :param ts: TensorSequence to be converted.
        :return: string representation of `ts`.
        """
        res = ""
        for i, index in enumerate(ts.indices):
            if i > 0:
                res += " + "
            coefficient = ts.array[i].squeeze()
            if np.all(np.isclose(coefficient.imag, 0)):
                coefficient = coefficient.real
            res += str(coefficient) + "*" + self.__alphabet.index_to_word(index)
        return res

    def print(self, ts: TensorSequence) -> None:
        """
        Prints an instance of TensorSequence as a string.

        :param ts: TensorSequence to be printed.
        """
        print(self.to_str(ts))

    @staticmethod
    def __2d_path_to_array(path: NDArray[float64], trunc: int) -> NDArray[float64]:
        """
        Converts a two-dimensional path into a two-dimensional array to be used to create the TensorSequence.

        :param path: A NumPy array of shape (len(t_grid), dim) representing the path,
            where rows correspond to time steps and columns to dimensions.
        :param trunc: The truncation level for the signature computation.
        :return: A TensorSequence representing the signature of the path up to the specified truncation level.
        """
        array = iisignature.sig(path, trunc, 2)
        array = np.vstack([np.zeros(array.shape[1]), array])
        array = np.hstack([np.ones((array.shape[0], 1)), array])
        array = array.T
        return array

    def path_to_sequence(self, path: NDArray[float64], trunc: int) -> TensorSequence:
        """
        Converts a path into a TensorSequence by computing its signature up to a given truncation level.

        :param path: A NumPy array of shape (len(t_grid), dim) representing the path,
            where rows correspond to time steps and columns to dimensions.
        :param trunc: The truncation level for the signature computation.
        :return: A TensorSequence representing the signature of the path up to the specified truncation level.
        """
        if path.ndim == 1:
            array = self.__2d_path_to_array(path=np.reshape(path, (1, -1)), trunc=trunc)
        elif path.ndim == 2:
            array = self.__2d_path_to_array(path=path, trunc=trunc)
        elif path.ndim == 3:
            array = np.zeros((2**(trunc + 1) - 1, path.shape[0], path.shape[2]))
            for i in range(path.shape[2]):
                array[:, :, i] = self.__2d_path_to_array(path=path[:, :, i], trunc=trunc)
        else:
            raise ValueError("Dimension of path should be less than 3.")
        return self.from_array(trunc=trunc, array=array)

    def plot_coefficients(self, ts: TensorSequence, trunc: int = None, nonzero: bool = False,
                          ax: plt.axis = None, **kwargs) -> None:
        """
        Plots the coefficients of a given tensor sequence.

        :param ts: tensor sequence to plot.
        :param trunc: truncation order, the coefficients of order <= trunc will be plotted. By default, equals to ts.trunc.
        :param nonzero: whether to plot only non-zero coefficients.
        :param ax: plt axis to plot on.
        """
        if trunc is None:
            trunc = ts.trunc

        n_coefficients = 2**(trunc + 1) - 1
        if nonzero:
            indices = ts.indices[ts.indices < n_coefficients]
            coefficients = ts.array[ts.indices < n_coefficients].squeeze().real
        else:
            indices = np.arange(n_coefficients)
            coefficients = np.zeros(n_coefficients)
            coefficients[ts.indices[ts.indices < n_coefficients]] = ts.array[ts.indices < n_coefficients].squeeze().real

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(coefficients, "o", **kwargs)
        ax.grid("on")
        ax.set_xticks(ticks=np.arange(coefficients.size),
                      labels=[self.alphabet.index_to_word(i) for i in indices],
                      rotation=-90)
        ax.plot()

    # TODO: Add truncation level for products.
    @staticmethod
    def tensor_prod(ts1: TensorSequence, ts2: TensorSequence) -> TensorSequence:
        """
        Computes the tensor product of two TensorSequences.
        :param ts1: The first TensorSequence.
        :param ts2: The second TensorSequence.
        :return: A new TensorSequence representing the tensor product of ts1 and ts2.
        """
        return ts1.tensor_prod(ts2)

    @staticmethod
    def shuffle_prod(ts1: TensorSequence, ts2: TensorSequence) -> TensorSequence:
        """
        Computes the shuffle product of two TensorSequences.
        :param ts1: The first TensorSequence.
        :param ts2: The second TensorSequence.
        :return: A new TensorSequence representing the shuffle product of ts1 and ts2.
        """
        return ts1.shuffle_prod(ts2)
