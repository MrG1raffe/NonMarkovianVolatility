from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numpy import int64, complex128
from typing import Union, Tuple
from numba.experimental import jitclass
import numba as nb

from signatures.alphabet import Alphabet
from signatures.shuffle import shuffle_product
from signatures.numba_utility import factorial

spec = [
    ('__alphabet', Alphabet.class_type.instance_type),
    ('__trunc', nb.int64),
    ('__array', nb.complex128[:, :, :]),
    ('__indices', nb.int64[:]),
]


@jitclass(spec)
class TensorSequence:
    def __init__(
        self,
        alphabet: Alphabet,
        trunc: int,
        array: NDArray[complex128],
        indices: NDArray[int64],
    ):
        """
        Initializes a TensorSequence object, which represents a collection of coefficients indexed
        by words from a specified alphabet, truncated at a certain length `trunc`.

        :param alphabet: An Alphabet object that defines the dimension and convertion functions.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
        :param array: Optional. A one-dimensional numpy array of tensor coefficients corresponding to the indices.
        :param indices: Optional. A numpy array of integer indices corresponding to words in the alphabet. For
            example, in the alphabet ("1", "2"), the word index is the index of word in the array
            ["Ø", "1", "2", "11", "12", "21", "22", "111", "112", "121", ...].
        # :param word_dict: Optional. A dictionary mapping words (as strings) to tensor coefficients.
        """
        self.__alphabet = alphabet
        self.__trunc = trunc

        if array.ndim == 1:
            reshaped_array = np.reshape(a=array, newshape=(-1, 1, 1)).astype(complex128)
        elif array.ndim == 2:
            reshaped_array = np.reshape(a=array, newshape=array.shape + (1,)).astype(complex128)
        elif array.ndim == 3:
            reshaped_array = array.astype(complex128)
        else:
            raise ValueError("Array can be at most 3-dimensional.")
        if len(indices):
            mask = indices < (self.__alphabet.dim**(trunc + 1) - 1) / (self.__alphabet.dim - 1)
            indices, reshaped_array = indices[mask], reshaped_array[mask]

        self.__indices = indices.astype(int64)
        self.__array = reshaped_array.astype(complex128)

        self.remove_zeros()

    def __getitem__(self, key: Union[str, int]) -> Union[complex128, NDArray[complex128], TensorSequence]:
        """
        Retrieves the coefficient associated with a given word if key is a string.
        If key is int, returns a tensor sequence corresponding in the time dimension (axis 1 of self.array).

        :param key: A string representing a word formed from the alphabet or integer representing the time index.
        :return: The tensor coefficient corresponding to the given word if type(key) == str
            or an instance of TensorSequence with the time index key if type(key) == int.
        """
        if isinstance(key, str):
            word_index = self.__alphabet.word_to_index(key)

            if word_index not in self.indices:
                return np.zeros((self.array.shape[1:]), dtype=complex128)

            array_index = np.searchsorted(self.indices, word_index)
            return self.__array[array_index]
        else:
            indexed_arr = np.ascontiguousarray(self.array[:, key, :])
            return TensorSequence(self.alphabet, self.trunc, indexed_arr, self.indices)

    def __rmul__(self, c: Union[float, complex, NDArray[complex128]]) -> TensorSequence:
        """
        Performs right multiplication of the TensorSequence by a scalar or a numpy array.

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        new_array = self.__array * c
        return TensorSequence(self.__alphabet, self.__trunc, new_array, self.__indices)

    def __mul__(self, c: Union[float, complex, NDArray[complex128]]) -> TensorSequence:
        """
        Performs left multiplication of the TensorSequence by a scalar or a numpy array
        of shape (self.__array.shape[0], 1).

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        return self.__rmul__(c)

    def __truediv__(self, c: Union[float, complex, NDArray[complex128]]):
        """
        Divides the TensorSequence by a scalar or numpy array of shape (self.__array.shape[0], 1).

        :param c: A scalar or numpy array by which to divide the TensorSequence.
        :return: A new TensorSequence that is the result of the division.
        """
        return self * (1 / c)

    @staticmethod
    def add_indices_and_arrays(indices_1: NDArray[complex128], array_1: NDArray[complex128],
                               indices_2: NDArray[complex128], array_2: NDArray[complex128]) -> Tuple[NDArray[complex128], NDArray[complex128]]:
        concatenated_indices = np.zeros(indices_1.size + indices_2.size)
        concatenated_indices[:indices_1.size] = indices_1
        concatenated_indices[indices_1.size:] = indices_2

        new_indices = np.unique(concatenated_indices)

        indices_first = np.searchsorted(new_indices, indices_1)
        indices_second = np.searchsorted(new_indices, indices_2)

        new_array = np.zeros((len(new_indices),) + array_1.shape[1:], dtype=complex128)
        new_array[indices_first] = new_array[indices_first] + array_1
        new_array[indices_second] = new_array[indices_second] + array_2
        return new_indices, new_array

    def __add__(self, ts: TensorSequence) -> TensorSequence:
        """
        Adds another TensorSequence to the current one.

        :param ts: The TensorSequence to add.
        :return: A new TensorSequence that is the result of the addition.
        """
        if not bool(self):
            return ts
        if not bool(ts):
            return self
        if self.__array.shape[1] != ts.array.shape[1]:
            raise ValueError("Time grids of sequences should be the same.")

        new_indices, new_array = self.add_indices_and_arrays(self.indices, self.array,
                                                             ts.indices, ts.array)
        return TensorSequence(self.__alphabet, self.__trunc, new_array, new_indices)

    def __sub__(self, ts: TensorSequence) -> TensorSequence:
        """
        Subtracts another TensorSequence from the current one.

        :param ts: The TensorSequence to subtract.
        :return: A new TensorSequence that is the result of the subtraction.
        """
        return self + ts * (-1.0)

    def __matmul__(self, ts: TensorSequence) -> Union[float, NDArray[complex128]]:
        """
        Computes the inner product (dot product) of the current TensorSequence with another.

        :param ts: The TensorSequence with which to compute the inner product.
        :return: The inner product as a scalar.
        """
        intersect_idx = np.intersect1d(self.__indices, ts.indices)
        sub_index_self = np.searchsorted(self.__indices, intersect_idx)
        sub_index_other = np.searchsorted(ts.indices, intersect_idx)
        return (self.array[sub_index_self] * ts.array[sub_index_other]).sum(axis=0)

    def __len__(self) -> int:
        """
        Returns the number of non-zero coefficients in the TensorSequence.

        :return: The number of non-zero coefficients.
        """
        return len(self.__indices)

    def __bool__(self) -> bool:
        """
        Returns whether the TensorSequence is non-empty (has non-zero coefficients).

        :return: True if the TensorSequence has non-zero coefficients, False otherwise.
        """
        return bool(len(self))

    def __zero(self) -> TensorSequence:
        """
        Creates an instance of TensorSequence with no indices and the same sizes of other axis.

        :return: A new zero TensorSequence.
        """
        self_shape = (0,) + self.array.shape[1:]
        return TensorSequence(self.__alphabet, self.__trunc, np.zeros(self_shape), np.zeros(0))

    def __unit(self) -> TensorSequence:
        """
        Creates an instance of TensorSequence with index 1 corresponding to the word Ø.

        :return: A unit element as TensorSequence.
        """
        self_shape = (1,) + self.array.shape[1:]
        return TensorSequence(self.__alphabet, self.__trunc, np.ones(self_shape), np.zeros(1))

    @staticmethod
    def zero(alphabet, trunc) -> TensorSequence:
        """
        Creates an instance of TensorSequence with no indices and the same sizes of other axis.

        :param alphabet: An Alphabet object that defines the dimension and convertion functions.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.

        :return: A new zero TensorSequence.
        """
        self_shape = (0, 1, 1)
        return TensorSequence(alphabet, trunc, np.zeros(self_shape), np.zeros(0))

    @staticmethod
    def unit(alphabet, trunc) -> TensorSequence:
        """
        Creates an instance of TensorSequence with index 1 corresponding to the word Ø.

        :param alphabet: An Alphabet object that defines the dimension and convertion functions.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.

        :return: A unit element as TensorSequence.
        """
        self_shape = (1, 1, 1)
        return TensorSequence(alphabet, trunc, np.ones(self_shape), np.zeros(1))

    def remove_zeros(self) -> None:
        """
        Removes elements of indices and array where the coefficients are equal to zero.
        """
        ATOL = 1e-12
        idx_to_keep = (np.abs(self.__array) > ATOL).sum(axis=2).sum(axis=1) > 0
        self.__indices = self.__indices[idx_to_keep]
        self.__array = self.__array[idx_to_keep]

    @property
    def alphabet(self) -> Alphabet:
        """
        Returns the array of tensor coefficients.

        :return: A numpy array of tensor coefficients.
        """
        return self.__alphabet

    @property
    def array(self) -> NDArray[complex128]:
        """
        Returns the array of tensor coefficients.

        :return: A numpy array of tensor coefficients.
        """
        return self.__array

    @property
    def trunc(self) -> int:
        """
        Returns the truncation level of the TensorSequence.

        :return: The truncation level as an integer.
        """
        return self.__trunc

    @property
    def indices(self) -> NDArray[int64]:
        """
        Returns the array of indices of words corresponding to non-zero coefficients of TensorSequence.

        :return: A numpy array of indices.
        """
        return self.__indices

    def update(self, ts: TensorSequence) -> None:
        """
        Updates the attributes of the instance copying the attributes of ts.

        :param ts: tensor sequence which attributes will be used as new attributes of self.
        """
        self.__alphabet = ts.alphabet
        self.__trunc = ts.trunc
        self.__array = ts.array
        self.__indices = ts.indices

    def norm_inf(self) -> NDArray[complex128]:
        """
        Computes the infinity norm (maximum absolute value) of the TensorSequence.

        :return: The infinity norm as a scalar or numpy array.
        """
        return np.abs(self.__array).sum(axis=0)

    def proj(self, word: str) -> TensorSequence:
        """
        Calculates the projection of TensorSequence with respect to the given word.

        :param word: The to calculate the projection.
        :return: A new TensorSequence representing the projection.
        """
        dim = self.__alphabet.dim
        word_index = self.__alphabet.word_to_index(word)
        indices_mask = (((self.indices - word_index) % dim**len(word)) == 0) & (self.indices >= word_index)
        indices_to_keep = self.indices[indices_mask]
        length_arr = self.__alphabet.index_to_length(indices_to_keep)
        new_indices = (indices_to_keep - dim**length_arr + 1) // dim**(len(word)) + dim**(length_arr - len(word)) - 1
        array = self.__array[indices_mask]
        return TensorSequence(self.__alphabet, self.__trunc, array, new_indices)

    def tensor_prod_word(self, word: str, coefficient: float = 1) -> TensorSequence:
        """
        Performs the tensor product of the current TensorSequence with a given word and
        multiply the result by `coefficient`.

        :param word: The word to tensor multiply with.
        :param coefficient: The coefficient to multiply the resulting tensor product.
        :return: A new TensorSequence representing the tensor product.
        """
        dim = self.__alphabet.dim
        word_dim_base = self.__alphabet.word_to_base_dim_number(word)
        length_indices = self.__alphabet.index_to_length(self.indices)
        new_indices = (dim**length_indices * dim**len(word) - 1) + \
            dim**len(word) * (self.indices - dim**length_indices + 1) + word_dim_base
        array = self.array * coefficient
        return TensorSequence(self.__alphabet, self.__trunc, array, new_indices)

    def tensor_prod_index(self, index: int, coefficient: Union[float, NDArray[complex128]] = 1) -> TensorSequence:
        """
        Performs the tensor product of the current TensorSequence with a given index and
        multiply the result by `coefficient`.

        :param index: The index of word to tensor multiply with.
        :param coefficient: The coefficient to multiply the resulting tensor product.
        :return: A new TensorSequence representing the tensor product.
        """
        dim = self.__alphabet.dim
        other_len = self.__alphabet.index_to_length(np.array([index], dtype=int64))
        other_dim_base = index - dim**other_len + 1
        length_indices = self.__alphabet.index_to_length(self.indices)
        new_indices = (dim**length_indices * dim**other_len - 1) + \
            dim**other_len * (self.indices - dim**length_indices + 1) + other_dim_base
        array = self.array * coefficient
        return TensorSequence(self.__alphabet, self.__trunc, array, new_indices)

    def tensor_prod(self, ts: TensorSequence) -> TensorSequence:
        """
        Performs the tensor product of the current TensorSequence with another TensorSequence.

        :param ts: The other TensorSequence to tensor multiply with.
        :return: A new TensorSequence representing the tensor product.
        """
        other_indices = ts.indices
        other_array = ts.array
        res = self.__zero()
        for i, other_index in enumerate(other_indices):
            coefficient = other_array[i]
            res.update(res + self.tensor_prod_index(other_index, coefficient))
        return res

    def shuffle_prod(self, ts: TensorSequence) -> TensorSequence:
        """
        Performs the shuffle product of the current TensorSequence with another TensorSequence.

        :param ts: The other TensorSequence to shuffle multiply with.
        :return: A new TensorSequence representing the shuffle product.
        """
        res_indices = np.zeros(0)
        res_array = np.zeros((0,) + self.array.shape[1:], dtype=complex128)

        words_self = [self.__alphabet.index_to_int(index) for index in self.indices]
        words_other = [self.__alphabet.index_to_int(index) for index in ts.indices]

        trunc = max(self.trunc, ts.trunc)

        for i_self, word_self in enumerate(words_self):
            for i_other, word_other in enumerate(words_other):
                if len(str(word_self)) + len(str(word_other)) <= trunc:
                    shuffle_words, counts = shuffle_product(word_self, word_other)
                    shuffle_indices = np.array([self.__alphabet.int_to_index(word) for word in shuffle_words], dtype=int64)
                    coefficients = self.array[i_self] * ts.array[i_other]
                    shuffle_array = (np.reshape(counts, (-1, 1, 1)) *
                                     np.reshape(coefficients, (1,) + coefficients.shape)).astype(complex128)

                    res_indices, res_array = self.add_indices_and_arrays(res_indices, res_array,
                                                                         shuffle_indices, shuffle_array)
        return TensorSequence(self.__alphabet, trunc, res_array, res_indices)

    def tensor_pow(self, p: int) -> TensorSequence:
        """
        Raises the TensorSequence to a tensor power p.

        :param p: The power to which the TensorSequence is raised.
        :return: A new TensorSequence representing the tensor power.
        """
        if p == 0:
            return self.__unit()

        res = self * 1
        # TODO: think about more efficient implementation (with log_2(p) operations)
        for _ in range(p - 1):
            res.update(res.tensor_prod(self))
        return res

    def shuffle_pow(self, p: int) -> TensorSequence:
        """
        Raises the TensorSequence to a shuffle power p.

        :param p: The power to which the TensorSequence is raised.
        :return: A new TensorSequence representing the shuffle power.
        """
        if p == 0:
            return self.__unit()

        res = self.__unit()
        # TODO: think about more efficient implementation (with log_2(p) operations)
        for _ in range(p):
            res.update(res.shuffle_prod(self))
        return res

    def tensor_exp(self, N_trunc: int) -> TensorSequence:
        """
        Computes the tensor exponential of the TensorSequence up to a specified truncation level.

        :param N_trunc: The truncation level for the exponential.
        :return: A new TensorSequence representing the tensor exponential.
        """
        res = self.__unit()
        for n in range(1, N_trunc):
            res.update(res + self.tensor_pow(n) / factorial(n))
        return res

    def shuffle_exp(self, N_trunc: int) -> TensorSequence:
        """
        Computes the shuffle exponential of the TensorSequence up to a specified truncation level.

        :param N_trunc: The truncation level for the exponential.
        :return: A new TensorSequence representing the shuffle exponential.
        """
        res = self.__unit()
        for n in range(1, N_trunc):
            res.update(res + self.shuffle_pow(n) / factorial(n))
        return res

    def resolvent(self, N_trunc):
        """
        Computes the resolvent of the TensorSequence up to a specified truncation level.
        The resolvent is defined as the series of the TensorSequence's tensor powers.

        :param N_trunc: The truncation level for the resolvent.
        :return: A new TensorSequence representing the resolvent.
        :raises ValueError: If the coefficient corresponding to the empty word exceeds or equals 1.
        """
        if np.max(np.abs(self[""])) >= 1:
            raise ValueError("Resolvent cannot be calculated. The tensor sequence l should have |l^∅| < 1.")
        res = self.__unit()
        for n in range(1, N_trunc):
            res.update(res + self.tensor_pow(n))
        return res
